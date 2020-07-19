#include "InputMLP.h"

InputMLP::InputMLP(int n_i, int n_h, int n_o){
    num_inp = n_i;
    num_h = n_h;
    num_osc = n_o;
    //std::cout << 'here';
    X_h = new float[num_h];
    X_osc = new float[num_osc];
    W1 = new float *[num_h];
    for(int i=0;i<num_h;i++){
        W1[i] = new float[num_inp];
        for(int j=0; j<num_inp; j++){
            W1[i][j] = get_random();
        }
    }
    W2 = new float *[num_osc];
    for(int i=0; i<num_osc;i++){
        W2[i] = new float[num_h];
        for(int j=0; j<num_h; j++){
            W2[i][j] = get_random();
        }
    }
}

void InputMLP::activationInput(float *X, float *out){
    float temp = 0.0; 
    for(int i=0; i<num_h; i++){
        temp = 0.0;
        for(int j=0; j<num_inp; j++){
            temp += W1[i][j]*X[j];
        }
        out[i] = tanh(temp);
    }    
}

void InputMLP::activationHidden(float *X, float *out){
    float temp = 0.0; 
    for(int i=0; i<num_osc; i++){
        temp = 0.0;
        for(int j=0; j<num_h; j++){
            temp += W2[i][j]*X[j];
        }   
        out[i] = tanh(temp);
    }     
}

void InputMLP::forwardPropagation(float *X, float *out){
    activationInput(X, X_h);
    activationHidden(X_h, X_osc);
    out = X_osc;
}
