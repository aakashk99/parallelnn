#include <math.h>
#include <cuda_runtime.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "util/img.h"
#include "neural/activations.h"
#include "neural/nn.h"
#include "matrix/matrix.h"
#include "matrix/ops.h"

int main() {
	//TIMING
	srand(time(NULL));
	cudaEvent_t start, stop;
    	cudaEventCreate(&start);
    	cudaEventCreate(&stop);
    	float elapsedTime;

	//TRAINING
	cudaEventRecord(start, 0);
	int number_imgs = 10000;
	Img** imgs = csv_to_imgs("./data/mnist_train.csv", number_imgs);
	NeuralNetwork* net = network_create(0.1, 3, 784, 300, 10);
	network_train_batch_imgs(net, imgs, number_imgs);
	network_save(net, "testing_net");

	cudaEventRecord(stop, 0);
    	cudaEventSynchronize(stop);
    	cudaEventElapsedTime(&elapsedTime, start, stop);
    	printf("Runtime: %3.1f ms\n", elapsedTime);

	// PREDICTING
//	cudaEventRecord(start, 0);
//	int number_imgs = 3000;
//	Img** imgs = csv_to_imgs("data/mnist_test.csv", number_imgs);
//	NeuralNetwork* net = network_load("testing_net");
//	double score = network_predict_imgs(net, imgs, number_imgs);
//	printf("Score: %1.5f\n", score);
//	cudaEventRecord(stop, 0);
//    	cudaEventSynchronize(stop);
//    	cudaEventElapsedTime(&elapsedTime, start, stop);
//    	printf("Runtime: %3.1f ms\n", elapsedTime);
//
//	imgs_free(imgs, number_imgs);
//	network_free(net);
    
	return 0;
}

