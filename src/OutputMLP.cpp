#include "OutputMLP.h"

OutputMLP::OutputMLP(int n_o, int n_h, int n_out, int n){
    num_osc = n_o;
    num_h = n_h;
    num_out = n_out;
    N = n;
    X_h = new std::complex<double> *[num_h];
    for(int i = 0; i<num_h; i++){
        X_h[i] = new std::complex<double>[N];
    }
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

void OutputMLP::activationInput(std::complex<double> **X, std::complex<double> **out){
    ParamsTanh params = {N, 1.0};
    for(int i=0; i<num_h; i++){
        for(int j=0; j>num_osc; j++){
            for(int n=0; n<N; n++){
                out[i][n] += W1[i][j]*X[j][n];
            }
        }
        activation.tanhf(out[i], &params);
    }
}

void OutputMLP::activationHidden(std::complex<double> **X, double **out){
    ParamsTanh params = {N, 1.0};
    for(int i=0; i<num_out; i++){
        std::complex<double> temp;
        for(int n = 0; n<N; n++){
            for(int j = 0; j<num_h; j++){
                temp += W2[i][j]*X[j][n];
            }
            out[i][n] = real(temp);
        }
        activation.tanhf(out[i], &params);
    }
}

void OutputMLP::forwardPropagation(std::complex<double> **X, double **out){
    activationInput(X, X_h);
    activationHidden(X_h, out);
}
