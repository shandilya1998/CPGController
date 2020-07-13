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

class OutputMLP{
    private:
        int num_osc;
        int num_h;
        int num_out;
        std::complex<float> **W1;
        std::complex<float> **W2;
        int N;
    public:   
        OutputMLP(int n_o, int n_h, int n_out, int n);
        void activationInput(std::complex<float> **X, std::complex<float> **out);
        void activationHidden(std::complex<float> **X, float **out);
        void forwardPropagation(std::complex<float> **X, float **out);
        void setInputWeights(std::complex<float> **weights){W1 = weights;}
        void setHiddenWeights(std::complex<float> **weights){W2 = weights;} 
};
