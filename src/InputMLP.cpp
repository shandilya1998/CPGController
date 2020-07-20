#include "InputMLP.h"

InputMLP::InputMLP(int n_i, int n_h, int n_o){
    num_inp = n_i;
    num_h = n_h;
    num_osc = n_o;
    //std::cout << 'here';
    X_h = new double[num_h];
    X_osc = new double[num_osc];
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

void InputMLP::activationInput(double *X, double *out){
    double temp = 0.0; 
    for(int i=0; i<num_h; i++){
        temp = 0.0;
        for(int j=0; j<num_inp; j++){
            temp += W1[i][j]*X[j];
        }
        out[i] = temp;
    }
    ParamsSigmoid params = {num_h, 1.0};
    activation.sigmoidf(out, &params);
}

void InputMLP::activationHidden(double *X, double *out){
    double temp = 0.0; 
    for(int i=0; i<num_osc; i++){
        temp = 0.0;
        for(int j=0; j<num_h; j++){
            temp += W2[i][j]*X[j];
        }   
        out[i] = temp;
    }     
    ParamsRelu params = {num_osc, 0.0};
    activation.reluf(out, &params);
}

void InputMLP::forwardPropagation(double *X, double *out){
    activationInput(X, X_h);
    activationHidden(X_h, X_osc);
    out = X_osc;
}
