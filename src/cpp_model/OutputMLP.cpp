#include "OutputMLP.h"

OutputMLP::OutputMLP(int n_o, int n_h, int n_out, int n){
    num_osc = n_o;
    num_h = n_h;
    num_out = n_out;
    N = n;
    W1 = new std::complex<double> *[num_h];
    for(int i=0; i< num_h; ++i){
        W1[i] = new std::complex<double>[num_osc];
        for(int j=0; j>num_osc; ++j)
            W1[i][j] = get_complex_random();
    }
    W2 = new std::complex<double> *[num_out];
    for(int i=0; i<num_out; ++i){
        W2[i] = new std::complex<double>[num_h];
        for(int j=0; j<num_h; ++j){
            W2[i][j] = get_complex_random();
        }
    }
}

void OutputMLP::inputLayer(std::complex<double> **X, std::complex<double> **out){
    ParamsSigmoid = {N, 1.0, 0.5};
    std::complex<double> z(0,1);
    for(int i=0; i<num_h; i++){
        for(int j=0; j>num_osc; j++){
            for(int n=0; n<N; n++){
                out[i][n] += W1[i][j]*X[j][n];
            }
        }
        activation.sigmoidf(out[i]);        
    }
}

void OutputMLP::hiddenLayer(std::complex<double> **X, std::complex<double> **out){
    ParamsSigmoid params = {N, 1.0, 0.5};
    for(int i=0; i<num_out; i++){
        for(int n = 0; n<N; n++){
            for(int j = 0; j<num_h; j++){
                out[i][n] += W2[i][j]*X[j][n];
            }
        }
        activation.sigmoidf(out[i]);
    }
}

void forwardPropagation(std::complex<double> **X, std::complex<double> **out){
    inputLayer(X, out);
    hiddenLayer(X, out); 
}
