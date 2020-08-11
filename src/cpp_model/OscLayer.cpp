#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, double dT){
    //omega = o;
    num_osc = n_o;
    dt = dT;
    N = n;
}

void OscLayer::computeZ(double freq, std::complex<double> *out, double *phi){
    std::complex<double> iota(0, 1);
    double r = new double[N];
    r[0] = 1.0;
    std::complex<double> i = -1;
    i = abs::sqrt(i);
    for(int i=1; i<N; i++){
        r[i] = r[i-1] + (1-pow(abs(r[i-1]),2))*r[i-1]*dt;
        phi[i] = phi[i-1] + freq*dt;
        out[i] = r[i]*std::exp(i*phi[i]);
    } 
}

void OscLayer::forwardPropagation(double *omega, std::complex<double> **Zout, double **phaseOut){
    for(int i=0; i<num_osc; i++){
        computeZ(omega[i], Zout[i], phaseOut[i]);  
    } 
}
