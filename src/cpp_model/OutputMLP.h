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
        std::complex<double> **W1;
        std::complex<double> **W2;
        int N;
        Activation activation;
    public:   
        OutputMLP(int n_o, int n_h, int n_out, int n);
        void inputLayer(std::complex<double> **X, std::complex<double> **out);
        void hiddenLayer(std::complex<double> **X, std::complex<double> **out);
        void forwardPropagation(std::complex<double> **X, std::complex<double> **out);
        void getInputWeights(std::complex<double> **weights){weights = W1;}
        void getHiddenWeights(std::complex<double> **weights){weights = W2;}
        void setInputWeights(std::complex<double> **weights){W1 = weights;}
        void setHiddenWeights(std::complex<double> **weights){W2 = weights;} 
};
