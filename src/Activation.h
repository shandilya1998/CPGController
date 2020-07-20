#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif

#ifndef COMPLEX
#define COMPLEX
#include <complex>
#endif

struct ParamsRelu{
    int dim;
    double weight;
};

struct ParamsSigmoid{
    int dim;
    double upperBound;
};

struct ParamsTanh{
    int dim;
    double bound;
};
class Activation{
    public:
        void reluf(double *inp, struct ParamsRelu*);
        void relugrad(double *inp, struct ParamsRelu*);
        void sigmoidf(double *inp, struct ParamsSigmoid*);
        void sigmoidgrad(double *inp, struct ParamsSigmoid*);
        void tanhf(double *inp, struct ParamsTanh*);
        void tanhgrad(double *inp, struct ParamsTanh*);
        void reluf(std::complex<double> *inp, struct ParamsRelu*);
        void relugrad(std::complex<double> *inp, struct ParamsRelu*);
        void sigmoidf(std::complex<double> *inp, struct ParamsSigmoid*);
        void sigmoidgrad(std::complex<double> *inp, struct ParamsSigmoid*);
        void tanhf(std::complex<double> *inp, struct ParamsTanh*);
        void tanhgrad(std::complex<double> *inp, struct ParamsTanh*);
};
