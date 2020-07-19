#include "DataLoader.h"


DataLoader::DataLoader(int num_osc, int n, int tsw, int tst, float *o, float a_h, float a_k){
    Tsw = tsw; 
    Tst = tst;
    T = Tsw+Tst;
    beta = (float) Tst/ (float) T;
    offset = o;
    A_h = a_h;
    A_k = a_k;
    num_osc = num_osc;
    N = n;
    heading = new float[2];
    heading[0] = 0;
    heading[1] = 1;
}

void DataLoader::HipJoint(float *out, int off, float h){
    int t;
    for(int j = T-off, k=0; j<N+T-off && k<N; j++,k++){
        t = j%T;
        if(t>=0 && t<T*beta/2){
            out[k] = A_h*sin(M_PI*(t/(T*beta)+M_PI))+h;
        }
        else if(t>=T*beta/2 && T*(2-beta)/2){
            out[k] = A_h*sin(M_PI*t/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta)))+h;
        }
        else{
            out[k] = A_h*sin(M_PI*t/(T*beta)+M_PI + M_PI*(beta-1)/beta)+h;
        }
    }
} 

void DataLoader::KneeJoint(float *out, int off){
    int t;
    for(int j = T-off, k = 0; j<N+T-off && k<N; j++, k++){
        t = j%T;
        if(t>=T*beta/2 && T*(2-beta)/2){
            out[k] = A_k*sin(M_PI*t/(T*(1-beta))+M_PI*beta/(2*(1-beta)))+90.0;
        }
        else{
            out[k] = 90.0;
        }
    }
}

void DataLoader::setHeading(float x, float y){
    heading[0] = x;
    heading[1] = y;
}

void DataLoader::getModelOutput(float **out){
    int off = T/4;
    float steer_angle;
    float temp;
    for(int i = 0; i< 2*num_osc; i++){
        if(i<4){
            steer_angle = atan(heading[1]/heading[0]);
            if(steer_angle>0){
                temp = (steer_angle)*180/(float)M_PI;
            }
            else{
                temp = (steer_angle+M_PI)*180/(float)M_PI;
            }
            HipJoint(out[i], i*off, temp);
        }
        else{
            KneeJoint(out[i], (i-4)*off);    
        }
    } 
}

void DataLoader::getModelInput(float *out){
    out = new float[3];
    out[0] = 2*6*A_h/T;
    out[1] = heading[0];
    out[2] = heading[1];
}
