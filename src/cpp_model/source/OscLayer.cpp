#include "OscLayer.h"

OscLayer::OscLayer(int n_o, int n, float dT){
    num_osc = n_o;
    dt = dT;
    N = n;
    r = 1;
    phi = 0;
    Z = new std::complex<float>*[num_osc];
    for(int i=0; i<num_osc; i++){
        Z[i] = new std::complex<float>[N];
    }
}

std::complex<float>** OscLayer::forwardPropagation(float *omega){
    std::complex<float> iota(0, 1); 
    float power = 2.0;
    float temp1, temp2;
    for(int i=0; i<num_osc; i++){
        for(int j =1; j<N; j++){
            r = r + (1-pow(abs(r), power))*r*dt;
            phi = phi + omega[i]*dt;
            Z[i][j] = r*cos(phi)+r*iota*sin(phi);
        }
        r = 1;
        phi = 0.0;  
    }
    /*
    for(int i=0; i<num_osc; i++){
        for(int j=0;j<N;j++){
            std::cout<<Z[i][j]<<"\t";
        }
        std::cout<<"\n";
    } 
    */
    return Z;
}
