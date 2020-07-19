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

#ifndef ACTIVATION
#define ACTIVATION
#include "Activation.h"
#endif

class OutputMLP{
    private:
        int num_osc;
        int num_h;
        int num_out;
        std::complex<float> **W1;
        std::complex<float> **W2;
        int N;
        std::complex<float> **X_h;
        float **X_out;
    public:   
        OutputMLP(int n_o, int n_h, int n_out, int n);
        void activationInput(std::complex<float> **X, std::complex<float> **out);
        void activationHidden(std::complex<float> **X, float **out);
        void forwardPropagation(std::complex<float> **X, float **out); 
        void getInputWeights(std::complex<float> **weights){weights = W1;}
        void getHiddenWeights(std::complex<float> **weights){weights = W2;}
        void getXh(std::complex<float> **x){x = X_h;}
        void getXout(float **x){x = X_out;}
};
