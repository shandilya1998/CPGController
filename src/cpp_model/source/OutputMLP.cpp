#include "OutputMLP.h"

OutputMLP::OutputMLP(int n_o, int n_h, int n_out, int n, float LR){
    num_osc = n_o;
    num_h = n_h;
    num_out = n_out;
    lr = LR;
    N = n;
    W1 = new std::complex<float> *[num_h];
    Z_h = new std::complex<float> *[num_h];
    Y_h = new std::complex<float> *[num_h];
    for(int i=0; i<num_h; i++){
        W1[i] = new std::complex<float> [num_osc];
        Z_h[i] = new std::complex<float> [N];
        Y_h[i] = new std::complex<float> [N];
        for(int j=0; j<num_osc; j++)
            W1[i][j] = get_complex_random();
    }
    Z_out = new std::complex<float> *[num_out];
    Y = new std::complex<float> *[num_out];
    W2 = new std::complex<float> *[num_out];
    for(int i=0; i<num_out; i++){
        W2[i] = new std::complex<float>[num_h];
        Z_out[i] = new std::complex<float>[N];
        Y[i] = new std::complex<float>[N];
        for(int j=0; j<num_h; j++){
            W2[i][j] = get_complex_random();
        }
    }
}

void OutputMLP::inputLayerOutput(std::complex<float> **X){
    ParamsSigmoid params = {N, 1.0, 0.5};
    std::complex<float> z(0,1);
    for(int i=0; i<num_h; i++){
        for(int j=0; j<num_osc; j++){
            for(int n=0; n<N; n++){
                Z_h[i][n] += W1[i][j]*X[j][n];
            }
        }
        activation.sigmoidf(Z_h[i], Y_h[i], params);        
    }
}

void OutputMLP::hiddenLayerOutput(std::complex<float> **X){
    ParamsSigmoid params = {N, 1.0, 0.5};
    for(int i=0; i<num_out; i++){
        for(int n = 0; n<N; n++){
            for(int j = 0; j<num_h; j++){
                Z_out[i][n] += W2[i][j]*X[j][n];
            }
        }
        activation.sigmoidf(Z_out[i], Y[i], params);
    }
}

std::complex<float>** OutputMLP::forwardPropagation(std::complex<float> **X){
    inputLayerOutput(X);
    hiddenLayerOutput(Y_h);
    return Y; 
}

void OutputMLP::backwardPropagation(float **signal, std::complex<float> **X){
    std::complex<float> iota(0,1);
    float temp1;
    float a = 0.5;
    for(int i=0; i<num_out; i++){
        for(int j=0; j<num_h; j++){ 
            for(int k=0; k<N; k++){
                temp1 = (
                    signal[i][k]-Y[i][k].real()
                )*a*(
                    1-Y[i][k].real()
                )*(
                    1+Y[i][k].real()
                );
                W2[i][j] -= -lr*temp1*(
                    Y_h[j][k].real()-iota*Y_h[j][k].imag()
                );
                for(int l=0; l<num_osc; l++){
                    W1[j][l] -= -lr*temp1*(
                        (
                            W2[i][j].real()*a*(
                                1-Y_h[j][k].real()
                            )*(
                                1+Y_h[j][k].real()
                            )*(
                                X[l][k].real() + iota*X[l][k].imag()
                            ) - W2[i][j].imag()*a*(
                                1-Y_h[j][k].imag()
                            )*(
                                1+Y_h[j][k].imag()
                            )*(
                                X[l][k].imag() + iota*X[l][k].real()
                            )
                        )
                    );
                }
            }
        }
    }
}
