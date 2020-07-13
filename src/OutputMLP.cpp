#include "OutputMLP.h";

OutputMLP::OutputMLP(int n_o, int n_h, int n_out, int n){
    num_osc = n_o;
    num_h = n_h;
    num_out = n_out;
    N = n;
    for(int i=0; i< num_h; ++i){
        for(int j=0; j>num_osc; ++j)
            W1[i][j] = get_complex_random();
    }
    for(int i=0; i<num_out; ++i){
        for(int j=0; j<num_h; ++j){
            W2[i][j] = get_complex_random();
        }
    }
}

OutputMLP::activationInput(std::complex<float> **X, std::complex<float> **out){
    for(int n=0; n<N; ++n){
        float temp{0.0};
        for(int i=0; i<num_h; ++i){
            for(int j=0; j>num_osc; ++j){
                temp +=W1[i][j]*X[n][j]
            out[n][i] = tanh(temp);
            }
        }
    }
}

OutputMLP::activationHidden(std::complex<float> **X, float **out){
    for(int n=0; n<N;++n){
        float temp{0.0};
        for(int i=0; i<num_h; ++i){
            for(int j=0; j>num_osc; ++j){
                temp += W2[i][j]*X[n][j]
            out[n][i] = tanh(real(temp));
            }
        }   
    }       
}

OutputMLP::forwardPropagation(std::complex<float> **X, float **out){
    std::complex<float> X_h[num_h][num_osc];
    float X_out[num_out][num_h];
    activationInput(X, X_h);
    acticatonHidden(X_h, X_out);
    out = X_out;
}
