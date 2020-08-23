#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, float dT){
    num_osc = n_o;
    dt = dT;
    N = n;
    r = new float[num_osc];
    phi = new float[num_osc];
    Z = new std::complex<float>*[num_osc];
    for(int i=0; i<num_osc; i++){
        Z[i] = new std::complex<float>[N];
        r[i] = 1.0;
        phi[i] = 0.0;
    }
}

std::complex<float>** OscLayer::forwardPropagation(float *omega){
    std::complex<float> iota(0, 1); 
    float power = 2.0;
    for(int i=0; i<N; i++){
        for(int j =0; j<num_osc; j++){
            r[j] = r[j] + (1-pow(abs(r[j]), power))*r[j]*dt;
            phi[j] = phi[j] + omega[j]*dt;
            Z[j][i] = r[i]*std::exp(phi[i]*iota);
        }  
    } 
    return Z;
}
