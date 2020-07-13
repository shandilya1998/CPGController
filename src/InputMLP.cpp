#include "InputMLP.h"

Input_MLP::Input_MLP(int n_i, int n_h, int n_o){
    num_inp = n_i;
    num_h = n_j;
    num_osc = n_o;
    for(int i=0;i<num_h;i++){
        for(int j=0; j<num_inp; j++){
            W1[i][j] = get_random();
        }
    }
    for(int i=0; i<num_osc;i++){
        for(int j=0; j<num_inp; j++){
            W2[i][j] = get_random();
        }
    }
}

Input_MLP::activationInput(float *X, float *out){
    float temp{0.0}; 
    for(int i=0; i<num_h; i++){
        temp = 0.0;
        for(int j=0; j<num_inp; j++){
            temp += W1[i][j]*X[j];
        }
        out[i] = tanh(temp);
    }    
}

Input_MLP::activationHidden(float *X, float *out){
    float temp{0.0}; 
    for(int i=0; i<num_osc; i++){
        temp = 0.0;
        for(int j=0; j<num_h; j++){
            temp += W2[i][j]*X[j];
        }   
        out[i] = tanh(temp);
    }     
}

Input_MLP::forwardPropagation(float *X, float *out){
    float X_h[num_h];
    float X_osc[num_osc];
    activationInput(X, X_h);
    activationHidden(X_h, X_osc)
    out = X_osc;
}

