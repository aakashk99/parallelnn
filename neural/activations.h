#ifndef __activations_H
#define __activations_H

#include "../matrix/matrix.h"

double sigmoid(double input);
Matrix* sigmoidPrime(Matrix* m);
Matrix* softmax(Matrix* m);

#endif
