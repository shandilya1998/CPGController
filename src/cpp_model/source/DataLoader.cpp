#include "DataLoader.h"

DataLoader::DataLoader(
    int n_o, 
    int n_out, 
    int num,
    float DT,
    int *tsw,
    int *tst,
    int *t,
    int n
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
    max = new float*[num_d];
    ff = new float[num_d];
    signals = new float**[num_d];
    norm_signals = new float**[num_d];
    mean = new float*[num_d];
    for(int i=0; i<num_d; i++){
        max[i] = new float[num_osc]();
        mean[i] = new float[num_osc]();
        beta[i] = (float)Tst[i]/(Tst[i]+Tsw[i]);
        signals[i] = new float *[num_osc];
        norm_signals[i] = new float *[num_osc];
        for(int j=0; j<num_osc; j++){
            signals[i][j] = new float[N]();
            norm_signals[i][j] = new float[N]();
        }
    }
    float zero = 0.0;
    input = new float[num_osc]();
    fourier.build(1/dt, N);
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
            for(int k=0; k<N+T; k++){
                t=k%T;
                pos = (int)(k-T*(1-(float)(j)/8));
                if(pos>=0 && pos<N){
                    if(t>=0 && t<(int)T*b/2){
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
                    mean[i][j] += signals[i][j][pos];
                    mean[i][j+1] += signals[i][j+1][pos];
                    if(abs(signals[i][j][pos])>max[i][j]){
                        max[i][j] = signals[i][j][pos];
                        //std::cout<<"hip signal: "<<signals[i][j][pos]<<"\n";
                        //std::cout<<"hip: "<<max[i][0]<<"\n";
                    }       
                    if(abs(signals[i][j+1][pos])>max[i][j+1]){
                        max[i][j+1] = signals[i][j+1][pos];
                        //std::cout<<"knee: "<<max[i][1]<<"\n";
                    } 
                }
            }
            mean[i][j] = mean[i][j]/N;
            mean[i][j+1] = mean[i][j+1]/N;
        }
        for(int j=0;j<num_osc; j=j+2){
            for(int k=0; k<N; k++){
                norm_signals[i][j][k] = (signals[i][j][k]-mean[i][j])/(1.2*max[i][j]);
                norm_signals[i][j+1][k] = (signals[i][j+1][k]-mean[i][j+1])/(1.2*max[i][j+1]);
                //std::cout<<norm_signals[i][j][k]<<"\n";
                //std::cout<<norm_signals[i][j+1][k]<<"\n";
            }   
        }
    }
}

void DataLoader::calcFF(){
    float p;
    for(int i=0; i<num_d; i++){
        ff[i] = fourier.fundamentalFrequency(signals[i][0]);
        std::cout<<"fundamental frequency for gait "+std::to_string(i)+": "<<ff[i]<<"\n";
    }
}

void DataLoader::setup(){
    createSignals();
    calcFF();
}
