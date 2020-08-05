#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, double dT){
    //omega = o;
    num_osc = n_o;
    dt = dT;
    N = n;
}

void OscLayer::computeZ(double freq, std::complex<double> *out, double *phi){
    std::complex<double> iota(0, 1);
    for(int i=1; i<N; i++){
        out[i] = out[i-1] + ((1-pow(abs(out[i-1]),2))*out[i-1] + iota*(freq*out[i-1]))*dt;
        phi[i] = phi[i-1] + freq*dt;
    } 
}

void OscLayer::forwardPropagation(double *omega, std::complex<double> **Zout, double **phaseOut){
    for(int i=0; i<num_osc; i++){
        computeZ(omega[i], Zout[i], phaseOut[i]);  
    } 
}
