#include "OutputMLP.h"

OutputMLP::OutputMLP(int n_o, int n_h, int n_out, int n){
    num_osc = n_o;
    num_h = n_h;
    num_out = n_out;
    N = n;
    X_h = new std::complex<float>*[num_h];
    X_out = new float*[num_out];
    W1 = new std::complex<float> *[num_h];
    for(int i=0; i< num_h; ++i){
        W1[i] = new std::complex<float>[num_osc];
        for(int j=0; j>num_osc; ++j)
            W1[i][j] = get_complex_random();
    }
    W2 = new std::complex<float> *[num_out];
    for(int i=0; i<num_out; ++i){
        W2[i] = new std::complex<float>[num_h];
        for(int j=0; j<num_h; ++j){
            W2[i][j] = get_complex_random();
        }
    }
}

void OutputMLP::activationInput(std::complex<float> **X, std::complex<float> **out){
    for(int n=0; n<N; ++n){
        std::complex<float> temp = 0.0;
        for(int i=0; i<num_h; ++i){
            for(int j=0; j>num_osc; ++j){
                temp +=W1[i][j]*X[n][j];
            out[n][i] = tanh(temp);
            }
        }
    }
}

void OutputMLP::activationHidden(std::complex<float> **X, float **out){
    for(int n=0; n<N;++n){
        std::complex<float> temp = 0.0;
        for(int i=0; i<num_h; ++i){
            for(int j=0; j>num_osc; ++j){
                temp += W2[i][j]*X[n][j];
            out[n][i] = tanh(real(temp));
            }
        }   
    }       
}

void OutputMLP::forwardPropagation(std::complex<float> **X, float **out){
    activationInput(X, X_h);
    activationHidden(X_h, X_out);
    out = X_out;
}
