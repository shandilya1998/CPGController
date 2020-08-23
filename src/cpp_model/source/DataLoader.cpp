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
    base = new float *[2];
    base[0] = new float[N];
    base[1] = new float[N];
    input = new float[num_osc];
}

void DataLoader::_populate(float **out, int index){
    int tst = Tst[index];
    int tsw = Tsw[index];
    int th = theta[index];
    float b = beta[index];
    float **base;
    int T = tst+tsw;
    for(int i=0;i<N+T; i++){
        int t = i%T;
        if(t>=0 && t<=T*b/2){
            base[0][i] = th*sin(M_PI*t/(T*b)+M_PI);
            base[1][i] = 0.0;
        } 
        else if(t>T*b/2 && t<=T*(2-b)/2){
            base[0][i] = th*sin(M_PI*t/(T*(1-b))+M_PI*(3-4*b)/(2*(1-b)));
            base[1][i] = th*sin(M_PI*t/(T*(1-b))-M_PI*b/(2*(1-b)));
        }
        else{
            base[0][i] = th*sin(M_PI*t/(T*b)+M_PI*(b-1)/b);
            base[1][i] = 0.0;
        }
    }
    int pos;
    for(int i=0; i<num_out; i=i+2){
        pos = i%2;
        for(int j=0; j<N; j++){
            out[i][j] = base[pos][j+T-i*T/8];
            out[i+1][j] = base[pos+1][j+T-i*T/8];
        }
    }
}

void DataLoader::createSignals(){
    tqdm bar;
    for(int i=0;i<num_d; i++){
        bar.progress(i, num_d);
        _populate(signals[i], i); 
    }
    bar.finish();
}

void DataLoader::calcFF(){
    std::vector<float> v(N);
    float pitch;
    for(int i=0; i<num_d; i++){
        for(int j=0; j<N; j++){
            v[j] = signals[i][0][j];
        }
        pitch = pitch::yin<float>(v, 1/dt);
        ff[i] = pitch;
    }
}

void DataLoader::setup(){
    createSignals();
    calcFF();
}
