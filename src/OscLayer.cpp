#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, float dT){
    //omega = o;
    num_osc = n_o;
    dt = dT;
    N = n;
    Z = new std::complex<float>[num_osc];
    for(int i=0; i<num_osc; i++){
        Z[i] = get_complex_random();
    }
}

void OscLayer::computeZ(float *freq, std::complex<float> *out){
    std::complex<float> iota = sqrt(-1);
    for(int i=0; i<num_osc; i++){
        out[i] += ((1-abs(Z[i]))*Z[i] + iota*(freq[i]*Z[i]))*dt;
        phi[i] += freq[i]*dt;
    } 
}

void OscLayer::forwardPropagation(float *omega, std::complex<float> **Zout, float **phaseOut){
    for(int i=0; i<N; i++){
        std::complex<float> out[num_osc];
        computeZ(omega, out);  
        Zout[i] = out;
        for(int i =0; i<num_osc; i++){
            Z[i] = out[i];
        }
        phaseOut[i] = phi;
    } 
}
