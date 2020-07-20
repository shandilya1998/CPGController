#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <cmath>
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
        double **W1;
        double **W2;
        double *X_h;
        double *X_osc;
        Activation activation;
    public:
        InputMLP(int n_i, int n_h, int n_o);
        void activationInput(double *X, double *out);
        void activationHidden(double *X, double *out);
        void forwardPropagation(double *X, double *out);        
        int getNumInputUnits(){return num_inp;}
        int getNumHiddenUnits(){return num_h;}
        int getNumOscUnits(){return num_osc;}        
        void getInputWeights(double **weights){weights = W1;}
        void getHiddenWeights(double **weights){weights = W2;}
        void getXh(double *x){x = X_h;}
        void getXosc(double *x){x = X_osc;}
};
