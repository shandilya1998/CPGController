#include "DataLoader.h"

DataLoader::DataLoader(
    int n_o, 
    int n_out, 
    int num,
    float DT,
    int *tsw,
    int *tst,
    int *t,
    int n,
    float cutoff
){
    num_osc = n_o;
    num_out = n_out;
    num_d = num;
    Tsw = tsw;
    Tst = tst;
    theta = t;
    dt = DT;
    N = n;
    beta = new float[num_d];
    for(int i=0; i<num_d; i++){
        beta[i] = Tst[i]/(Tst[i]+Tsw[i]);
    }
    ff = new float[num];
    signals = new float**[num];
    for(int i=0; i<num; i++){
        signals[i] = new float *[num_osc];
        for(int j=0; j<num_osc; j++){
            signals[i][j] = new float[N];
        }
    }
    float zero = 0.0;
    //sig.assign(N, 0);
    sig.reserve(N);
    for(int i=0; i<N;i++){
        sig.emplace_back(i);
    }
    input = new float[num_osc];
}

void DataLoader::createSignals(){
    int tst;
    int tsw;
    int th;
    float b;
    int T;
    int t;
    int pos;
    for(int i=0;i<num_d; i++){
        tst = Tst[i];
        tsw = Tsw[i];
        th = theta[i];
        b = beta[i];
        T = tst+tsw;
        for(int j=0;j<num_osc; j=j+2){
            for(int k=T; k<N+T; k++){
                t=k%T;
                pos = (int)(k-T*(1-(float)(j/8)));
                if(pos>0){
                    if(t>=0 && t<=T*b/2){
                        signals[i][j][pos] = th*sin(M_PI*t/(T*b)+M_PI);
                        signals[i][j+1][pos] = 0.0;
                    }
                    else if(t>T*b/2 && t<=T*(2-b)/2){
                        signals[i][j][pos] = th*sin(M_PI*t/(T*(1-b))+M_PI*(3-4*b)/(2*(1-b)));
                        signals[i][j+1][pos] = th*sin(M_PI*t/(T*(1-b))-M_PI*b/(2*(1-b)));
                    }
                    else if(t>T*(2-b)/2 && t<=T){
                        signals[i][j][pos] = th*sin(M_PI*t/(T*b)+M_PI*(b-1)/b);
                        signals[i][j+1][pos] = 0.0;
                    }
                }
            } 
        }
    }
}

void DataLoader::calcFF(std::vector<float> &vec){
    float p;
    for(int i=0; i<num_d; i++){
        for(int j=0; j<N; j++){
            vec.at(j) = signals[i][0][j];
        }
        //p = pitch::mpm<float>(vec, (int)(1/dt));
        ff[i] = p;
    }
}

void DataLoader::setup(){
    createSignals();
    calcFF(sig);
}
