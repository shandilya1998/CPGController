#include "DataLoader.h"

DataLoader::DataLoader(int num_osc, int n, float tsw, float tst, float *phase, float *a){
    Tsw = tsw; 
    Tst = tst;
    T = Tsw+Tst;
    beta = Tst/T;
    phaseOffset = phase;
    A = a;
    num_osc = num_osc;
    N = n;
}

void HipJoint(float **out){
    for(int i = 0; i<num_osc; i++){
        for(int j = 0; j<N; j++){
            t = N%T;
            if(t>=0 && t<T*beta/2){
                out[i][j] = A[i]*sin(M_PI*t/(T*beta)+M_PI+phaseOffset[i])
            }
        }
    }
} 
