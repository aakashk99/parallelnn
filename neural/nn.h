#ifndef __nn_H
#define __nn_H

#include "../matrix/matrix.h"
#include "../util/img.h"

typedef struct {
	int layers;
	double learning_rate;
	int* nodes;
	Matrix** weights;
} NeuralNetwork;

NeuralNetwork* network_create(double lr, int layers, ...);
int network_train(NeuralNetwork* net, Matrix* input_data, Matrix* output_data, int label);
void network_train_batch_imgs(NeuralNetwork* net, Img** imgs, int batch_size);
Matrix* network_predict_img(NeuralNetwork* net, Img* img);
double network_predict_imgs(NeuralNetwork* net, Img** imgs, int n);
Matrix* network_predict(NeuralNetwork* net, Matrix* input_data);
void network_save(NeuralNetwork* net, char* file_string);
NeuralNetwork* network_load(char* file_string);
void network_print(NeuralNetwork* net);
void network_free(NeuralNetwork* net);

#endif
