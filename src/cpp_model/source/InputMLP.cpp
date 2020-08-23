#include "InputMLP.h"

InputMLP::InputMLP(int n_i, int n_h, int n_o){
    num_inp = n_i;
    num_h = n_h;
    num_osc = n_o;
    //std::cout << 'here';
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
    Y_h = new float[num_h];
    Y_osc = new float[num_osc];
}

void InputMLP::inputLayerOutput(float *X){
    float temp = 0.0; 
    ParamsSigmoid params = {
        num_h, 
        1,
        0.5,
    };
    for(int i=0; i<num_h; i++){
        for(int j=0; j<num_inp; j++){
            Z_h[i] += W1[i][j]*X[j];
        }
        activation.sigmoidf(Z_h, Y_h, params);        
    }
    delete params
}

void InputMLP::hiddenLayerOutput(float *X){
    float temp = 0.0; 
    ParamsSigmoid params = { 
        num_h, 
        1,  
        0.5,
    };
    for(int i=0; i<num_osc; i++){
        for(int j=0; j<num_h; j++){
            Z_osc[i] += W2[i][j]*X[j];
        }
        activation.sigmoidf(Z_osc, Y_osc, params)   
    }     
}

float* InputMLP::forwardPropagation(float *X){
    inputLayerOutput(X);
    hiddenLayerOutput(X);
    return Y_osc;
}
