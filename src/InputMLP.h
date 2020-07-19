#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS

#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif

class InputMLP{
    private:
        int num_inp;
        int num_h;
        int num_osc;
        float **W1;
        float **W2;
        float *X_h;
        float *X_osc;
    public:
        InputMLP(int n_i, int n_h, int n_o);
        void activationInput(float *X, float *out);
        void activationHidden(float *X, float *out);
        void forwardPropagation(float *X, float *out);        
        int getNumInputUnits(){return num_inp;}
        int getNumHiddenUnits(){return num_h;}
        int getNumOscUnits(){return num_osc;}        
        void getInputWeights(float **weights){weights = W1;}
        void getHiddenWeights(float **weights){weights = W2;}
        void getXh(float *x){x = X_h;}
        void getXosc(float *x){x = X_osc;}
};
