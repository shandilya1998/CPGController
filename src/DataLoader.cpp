#include "DataLoader.h"


DataLoader::DataLoader(int num_osc, int n, int tsw, int tst, double *o, double a_h, double a_k){
    Tsw = tsw; 
    Tst = tst;
    T = Tsw+Tst;
    beta = (double) Tst/ (double) T;
    offset = o;
    A_h = a_h;
    A_k = a_k;
    num_osc = num_osc;
    N = n;
    heading = new double[2];
    heading[0] = 0;
    heading[1] = 1;
}

void DataLoader::HipJoint(double *out, int off, double h){
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

void DataLoader::KneeJoint(double *out, int off){
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

void DataLoader::setHeading(double x, double y){
    heading[0] = x;
    heading[1] = y;
}

void DataLoader::getModelOutput(double **out){
    int off = T/4;
    double steer_angle;
    double temp;
    for(int i = 0; i< 2*num_osc; i++){
        if(i<4){
            steer_angle = atan(heading[1]/heading[0]);
            if(steer_angle>0){
                temp = (steer_angle)*180/(double)M_PI;
            }
            else{
                temp = (steer_angle+M_PI)*180/(double)M_PI;
            }
            HipJoint(out[i], i*off, temp);
        }
        else{
            KneeJoint(out[i], (i-4)*off);    
        }
    } 
}

void DataLoader::getModelInput(double *out){
    out = new double[3];
    out[0] = 2*6*A_h/T;
    out[1] = heading[0];
    out[2] = heading[1];
}
