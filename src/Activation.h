#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS

#include <iostream>
#include <cmath>
#include "random_num_generator.h"

#endif

class Activation{
    public:
        void reluf(float *inp, float *out, float *params);
        void relugrad(float *inp, float *out, float *params);
        void sigmoidf(float *inp, float *out, float *params);
        void sigmoidgrad(float *inp, float *out, float *params);
        void tanhf(float *inp, float *out, float *params);
        void tanhgrad(float *inp, float *out, float *params);
};
