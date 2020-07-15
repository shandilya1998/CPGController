#include "DataLoader.h"

DataLoader::DataLoader(int num_osc, int n, float tsw, float tst, float *phase, float *a_h, float *a_k){
    Tsw = tsw; 
    Tst = tst;
    T = Tsw+Tst;
    beta = Tst/T;
    phaseOffset = phase;
    A_h = a_h;
    A_k = a_k;
    num_osc = num_osc;
    N = n;
}

void DataLoader::HipJoint(float **out){
    for(int i = 0; i<num_osc; i++){
        for(int j = 0; j<N; j++){
            t = N%T;
            if(t>=0 && t<T*beta/2){
                out[i][j] = A_h[i]*sin(M_PI*t/(T*beta)+M_PI + phaseOffset[i])
            }
            else if(t>=T*beta/2 && T*(2-beta)/2){
                out[i][j] = A_h[i]*sin(M_PI*t/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta)) + phaseOffset[i])
            }
            else{
                out[i][j] = A_h[i]*sisn(M_PI*t/(T*beta)+M_PI + M_PI*(beta-1)/beta + phaseOffset[i])
            }
        }
    }
} 

void DataLoader::KneeJoint(float **out){
    for(int i = 0; i<num_osc; i++){
        for(int j = 0; j<N; j++){
            t = N%T;
            if(t>=T*beta/2 && T*(2-beta)/2){
                out[i][j] = A_k[i]*sin(M_PI*t/(T*(1-beta))+M_PI*beta/(2*(1-beta)))
            }
            else{
                out[i][j] = 0.0;
            }
        }
    }
}
