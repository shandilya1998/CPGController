#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <math.h>
#include "random_num_generator.h"
#endif

#ifndef ACTIVATION
#define ACTIVATION
#include "Activation.h"
#endif

class InputMLP{
    private:
        int num_inp;
        int num_h;
        int num_osc;
        Activation activation;
        float **W1;
        float **W2;
        float *Y_h;
        float *Y_osc;
        float *Z_h;
        float *Z_osc;
        void inputLayerOutput(float *X);
        void hiddenLayerOutput(float *X);
    public:
        InputMLP(int n_i, int n_h, int n_o);
        int getNumInputUnits(){return num_inp;}
        int getNumHiddenUnits(){return num_h;}
        int getNumOscUnits(){return num_osc;}        
        float** getInputWeights(){return W1;}
        float** getHiddenWeights(){return W2;}
        float* forwardPropagation(float *X);
};
