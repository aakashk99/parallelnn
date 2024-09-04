#include "nn.h"
#include <sys/stat.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdarg.h>
#include "../matrix/ops.h"
#include "activations.h"

#define MAXCHAR 1000

// 784, 300, 10
NeuralNetwork* network_create(double lr, int layers, ...) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	net->layers = layers;
	net->learning_rate = lr;

	int* nodes = malloc((sizeof(int)) * layers);
	va_list args;
	va_start(args, layers);
	for (int i = 0; i < layers; ++i) {
		nodes[i] = va_arg(args, int);	
	}
	va_end(args);
	net->nodes = nodes;

	Matrix** weights = malloc((sizeof(Matrix*)) * (layers - 1));
	for (int i = 0; i < layers - 1; ++i) {
		weights[i] = malloc(sizeof(Matrix));	
		Matrix* layer = matrix_create(net->nodes[i+1], net->nodes[i]);
		matrix_randomize(layer, net->nodes[i+1]);
		weights[i] = layer;
	}
	net->weights = weights;
}

int network_train(NeuralNetwork* net, Matrix* input, Matrix* output, int label) {
	// Feed forward
	Matrix** outputs = malloc((sizeof(Matrix*)) * (net->layers - 1));
	outputs[0] = malloc(sizeof(Matrix));
	outputs[0] = apply(sigmoid, dot(net->weights[0], input));
	for (int i = 1; i < net->layers - 1; ++i) {
		outputs[i] = malloc(sizeof(Matrix));
		outputs[i] = apply(sigmoid, dot(net->weights[i], outputs[i-1]));
	}

	int res = 0;
	if (matrix_argmax(outputs[net->layers - 2]) == label) res = 1;

	// Find errors
	Matrix** errors = malloc((sizeof(Matrix*)) * (net->layers - 1));
	errors[net->layers - 2] = malloc(sizeof(Matrix));
	errors[net->layers - 2] = subtract(output, outputs[net->layers - 2]);	
	for (int i = net->layers - 3; i >= 0; --i) {
		errors[i] = malloc(sizeof(Matrix));
		Matrix* transposed = transpose(net->weights[i+1]);
		errors[i] = dot(transposed, errors[i+1]);
		matrix_free(transposed);
	}

	// Backpropagate
	Matrix* sigmoid_primed_mat;
	Matrix* multiplied_mat;
	Matrix* dot_mat;
	Matrix* scaled_mat;
	Matrix* added_mat;
	Matrix* transposed_mat;
	for (int i = net->layers - 2; i >= 1; --i) {
		sigmoid_primed_mat = sigmoidPrime(outputs[i]); 	
		multiplied_mat = multiply(errors[i], sigmoid_primed_mat);
		transposed_mat = transpose(outputs[i - 1]);
		dot_mat = dot(multiplied_mat, transposed_mat);
		scaled_mat = scale(net->learning_rate, dot_mat);
		added_mat = add(net->weights[i], scaled_mat);

		matrix_free(net->weights[i]);
		net->weights[i] = added_mat;

		matrix_free(sigmoid_primed_mat);
		matrix_free(multiplied_mat);
		matrix_free(transposed_mat);
		matrix_free(dot_mat);
		matrix_free(scaled_mat);
	}	

	sigmoid_primed_mat = sigmoidPrime(outputs[0]);
	multiplied_mat = multiply(errors[0], sigmoid_primed_mat);
	transposed_mat = transpose(input);
	dot_mat = dot(multiplied_mat, transposed_mat);
	scaled_mat = scale(net->learning_rate, dot_mat);
	added_mat = add(net->weights[0], scaled_mat);
	matrix_free(net->weights[0]); 
	net->weights[0] = added_mat; 

	matrix_free(sigmoid_primed_mat);
	matrix_free(multiplied_mat);
	matrix_free(transposed_mat);
	matrix_free(dot_mat);
	matrix_free(scaled_mat);


	// Free matrices
	for (int i = 0; i < net->layers - 1; ++i) {
		matrix_free(outputs[i]);	
	}	
	free(outputs);
	outputs = NULL;
	for(int i = 0; i < net->layers - 1; ++i) {
		matrix_free(errors[i]);	
	}
	free(errors);
	errors = NULL; 

	return res;
}

void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size) {
	int correct = 0;
	for (int i = 0; i < batch_size; i++) {
		if (i % 100 == 0) printf("Img No. %d\n", i);
		if (i % 100 == 0) printf("Training Accuracy: %.6f\n", (double) correct/i);
		Img* cur_img = imgs[i];
		Matrix* img_data = matrix_flatten(cur_img->img_data, 0); // 0 = flatten to column vector
		Matrix* output = matrix_create(10, 1);
		matrix_fill(output, 0);
		output->entries[cur_img->label] = 1; // Setting the result
		correct += network_train(net, img_data, output, cur_img->label);
		matrix_free(output);
		matrix_free(img_data);
	}
}

Matrix* network_predict_img(NeuralNetwork* net, Img* img) {
	Matrix* img_data = matrix_flatten(img->img_data, 0);
	Matrix* res = network_predict(net, img_data);
	matrix_free(img_data);
	return res;
}

double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n) {
	int n_correct = 0;
	for (int i = 0; i < n; i++) {
		Matrix* prediction = network_predict_img(net, imgs[i]);
		img_print(imgs[i]);
		printf("Output Vector: ");
		for (int i = 0; i < 10; ++i)
			printf("%.6f ", prediction->entries[i]);
		printf("\n");
		printf("Network Prediction: %d\n", matrix_argmax(prediction));
		if (matrix_argmax(prediction) == imgs[i]->label) {
			n_correct++;
		}
		matrix_free(prediction);
	}
	return 1.0 * n_correct / n;
}

Matrix* network_predict(NeuralNetwork* net, Matrix* input_data) {
	Matrix* cur_input= dot(net->weights[0], input_data);
	Matrix* cur_output = apply(sigmoid, cur_input);
	for (int i = 1; i < net->layers - 1; ++i) {
		cur_input = dot(net->weights[i], cur_output);
		cur_output = apply(sigmoid, cur_input);
	}

	Matrix* result = softmax(cur_output);

	matrix_free(cur_input);
	matrix_free(cur_output);

	return result;
}

void network_save(NeuralNetwork* net, char* file_string) {
	mkdir(file_string, 0777);
	// Write the descriptor file
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "w");
	fprintf(descriptor, "%d\n", net->layers);
	for (int i = 0; i < net->layers; ++i) {
		fprintf(descriptor, "%d\n", net->nodes[i]);
	}
	fclose(descriptor);

	//Saving Weight Matrices
	char buffer[MAXCHAR];
	for (int i = 0; i < net->layers - 1; ++i) {
		snprintf(buffer, MAXCHAR, "Weight Matrix %d", i);
		matrix_save(net->weights[i], buffer); 
	}
	printf("Successfully written to '%s'\n", file_string);
	chdir("-"); // Go back to the orignal directory
}

NeuralNetwork* network_load(char* file_string) {
	NeuralNetwork* net = malloc(sizeof(NeuralNetwork));
	char entry[MAXCHAR];
	chdir(file_string);
	FILE* descriptor = fopen("descriptor", "r");
	fgets(entry, MAXCHAR, descriptor);
	net->layers = atoi(entry);
	int* nodes = malloc((sizeof(int)) * net->layers);
	for (int i = 0; i < net->layers; ++i) {
		fgets(entry, MAXCHAR, descriptor);
		nodes[i] = atoi(entry);
	}
	net->nodes = nodes;
	fclose(descriptor);

	Matrix** weights = malloc((sizeof(Matrix*)) * (net->layers - 1));
	char buffer[MAXCHAR];
	for (int i = 0; i < net->layers - 1; ++i) {
		weights[i] = malloc(sizeof(Matrix));
		snprintf(buffer, MAXCHAR, "Weight Matrix %d", i);
		weights[i] = matrix_load(buffer);
	}
	net->weights = weights;
	printf("Successfully loaded network from '%s'\n", file_string);
	chdir("-"); // Go back to the original directory
	return net;
}

void network_print(NeuralNetwork* net) {
	printf("Network Topology: \n");
	for (int i = 0; i < net->layers; ++i) {
		printf("Layer #%d: %d nodes\n", i, net->nodes[i]);	
	}
	for (int i = 0; i < net->layers - 1; ++i) {
		printf("Weight Matrix %d: \n", i);	
		matrix_print(net->weights[i]);
	}
}

void network_free(NeuralNetwork *net) {
	for (int i = 0; i < net->layers - 1; ++i) {
		matrix_free(net->weights[i]);
	}
	free(net->weights);
	free(net->nodes);
	free(net);
	net = NULL;
}
