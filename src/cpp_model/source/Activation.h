#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <math.h>
#include "random_num_generator.h"
#endif

#ifndef COMPLEX
#define COMPLEX
#include <complex>
#endif

struct ParamsRelu{
    int dim;
    float weight;
};

struct ParamsSigmoid{
    int dim;
    float upperBound;
    float weight;
};

struct ParamsTanh{
    int dim;
    float bound;
};
class Activation{
    public:
        void sigmoidf(float *inp, float *out, struct ParamsSigmoid);
        void sigmoidgrad(float *inp, float *out, struct ParamsSigmoid);
        void sigmoidf(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid);
        void sigmoidgrad(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid);
};
