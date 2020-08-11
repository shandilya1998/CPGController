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
    double weight;
};

struct ParamsTanh{
    int dim;
    double bound;
};
class Activation{
    public:
        void reluf(double *inp, double *out, struct ParamsRelu*);
        void relugrad(double *inp, double *out, struct ParamsRelu*);
        void sigmoidf(double *inp, double *out, struct ParamsSigmoid*);
        void sigmoidgrad(double *inp, double *out, struct ParamsSigmoid*);
        void tanhf(double *inp, double *out, struct ParamsTanh*);
        void tanhgrad(double *inp, double *out, struct ParamsTanh*);
        void reluf(std::complex<double> *inp, double *out, struct ParamsRelu*);
        void relugrad(std::complex<double> *inp, double *out, struct ParamsRelu*);
        void sigmoidf(std::complex<double> *inp, double *out, struct ParamsSigmoid*);
        void sigmoidgrad(std::complex<double> *inp, double *out, struct ParamsSigmoid*);
        void tanhf(std::complex<double> *inp, double *out, struct ParamsTanh*);
        void tanhgrad(std::complex<double> *inp, double *out, struct ParamsTanh*);
};
