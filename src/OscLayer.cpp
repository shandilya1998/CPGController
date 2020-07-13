#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, float dT){
    //omega = o;
    num_osc = n_o;
    dt = dT;
    N = n;
    for(int i=0; i<num_osc; i++){
        Z[i] = get_complex_random();
    }
}

OscLayer::computeZ(float *freq){
    for(int i=0; i<num_osc; i++){
        Z[i] += ((1-abs(Z[i]))*Z[i] + 1i*freq[i]*Z[i])*dt 
        phi[i] += freq[i]*dt 
    } 
}

OscLayer::forwardPropagation(float *omega, std::complex<float> **Zout, float **phaseOut){
    for(int i=0; i<N; i++){
        computeZ(omega);  
        Zout[i] = Z; 
        phaseOut[i] = phi;
    } 
}
