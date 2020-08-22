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
        float lr;
        std::complex<float> **W1;
        std::complex<float> **W2;
        int N;
        std::complex<float> **Z_h;
        std::complex<float> **Z_out;
        std::complex<float> **Y_h;
        std::complex<float> **Y;
        Activation activation;
        void inputLayerOutput(std::complex<float> **X);
        void hiddenLayerOutput(std::complex<float> **X);
    public:
        ~OutputMLP(){
            delete W1;
            delete ~W2;
            delete Z_h;
            delete Z_out;
            delete Y_h;
            delete Y;
        }   
        OutputMLP(int n_o, int n_h, int n_out, int n, float LR);
        std::complex<float>** forwardPropagation(std::complex<float> **X);
        std::complex<float> getInputWeights(std::complex<float> **weights){return W1;}
        std::complex<float> getHiddenWeights(std::complex<float> **weights){return W2;}
        void setInputWeights(std::complex<float> **weights){W1 = weights;}
        void setHiddenWeights(std::complex<float> **weights){W2 = weights;} 
        void backwardPropagation(float **signal, float **X);
};
