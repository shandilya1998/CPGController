#include "InputMLP.h"

InputMLP::InputMLP(int n_i, int n_h, int n_o){
    num_inp = n_i;
    num_h = n_h;
    num_osc = n_o;
    //std::cout << 'here';
    W1 = new double *[num_h];
    for(int i=0;i<num_h;i++){
        W1[i] = new double[num_inp];
        for(int j=0; j<num_inp; j++){
            W1[i][j] = get_random();
        }
    }
    W2 = new double *[num_osc];
    for(int i=0; i<num_osc;i++){
        W2[i] = new double[num_h];
        for(int j=0; j<num_h; j++){
            W2[i][j] = get_random();
        }
    }
}

void InputMLP::weightedSumInput(double *X, double *out){
    double temp = 0.0; 
    for(int i=0; i<num_h; i++){
        for(int j=0; j<num_inp; j++){
            out[i] += W1[i][j]*X[j];
        }
    }
}

void InputMLP::weightedSumHidden(double *X, double *out){
    double temp = 0.0; 
    for(int i=0; i<num_osc; i++){
        for(int j=0; j<num_h; j++){
            out[i] += W2[i][j]*X[j];
        }   
    }     
}
