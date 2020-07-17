#include "DataLoader.h"


DataLoader::DataLoader(int num_osc, int n, float tsw, float tst, float *o, float *a_h, float *a_k){
    Tsw = tsw; 
    Tst = tst;
    T = Tsw+Tst;
    beta = Tst/T;
    offset = o;
    A_h = a_h;
    A_k = a_k;
    num_osc = num_osc;
    N = n;
    heading = new float[2];
    heading = {0, 1};
}

void DataLoader::HipJoint(float *out, int i, int off){
    for(int j = T-off, k=0; j<N+T-off && k<N; j++,k++){
        t = j%T;
        if(t>=0 && t<T*beta/2){
            out[k] = A_h[i]*sin(M_PI*(t/(T*beta)+M_PI)+offset[i];
        }
        else if(t>=T*beta/2 && T*(2-beta)/2){
            out[k] = A_h[i]*sin(M_PI*t/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta)))+offset[i];
        }
        else{
            out[k] = A_h[i]*sisn(M_PI*t/(T*beta)+M_PI + M_PI*(beta-1)/beta)+offset[i];
        }
    }
} 

void DataLoader::KneeJoint(float *out, int i, int off){
    for(int j = T-off, k = 0; j<N+T-off && k<N; j++, k++){
        t = j%T;
        if(t>=T*beta/2 && T*(2-beta)/2){
            out[k] = A_k[i]*sin(M_PI*t/(T*(1-beta))+M_PI*beta/(2*(1-beta)))
        }
        else{
            out[k] = 0.0;
        }
    }
}

void DataLoader::setHeading(float x, float y){
    heading[0] = x;
    heading[1] = y;
}

void DataLoader::getData(float **out){
    int off = T/4;
    float steer_angle;
    for(int i = 0; i< num_osc; i++){
        if(i<4){
            steer_angle = atan(y/x);
            if(steer_angle>0){
                out[i] += (steer_angle)*180/M_PI;
            }
            else{
                out[i] += (steer_angle+M_PI)*180/M_PI;
            }
            HipJoint(out[i], i, i*off); 
            if(steer_angle>0){
                out[i] += (steer_angle)*180/M_PI;
            }   
            else{
                out[i] += (steer_angle+M_PI)*180/M_PI;
            }
        }
        else{
            KneeJoint(out[i], i-4, (i-4)*off);    
        }
    } 
}
