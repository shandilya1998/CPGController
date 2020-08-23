//#include "InputMLP.h"
#include "OscLayer.h"
#include "OutputMLP.h"
#include "DataLoader.h"
#ifndef COMPLEX
#define COMPLEX
#include <complex>
#endif
#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif
#ifndef TQDM
#define TQDM
#include "tqdm.h"
#endif
#include "matplotlibcpp.h"
#include <string>

namespace plt = matplotlibcpp;

float mse(float **Y, float **Y_p, int n1, int n2){
    float s=0;
    for(int i=0;i<n1;i++){
        for(int j=0; j<n2;j++){
            s += pow(Y[i][j]-Y_p[i][j], 2);
        }
    }
    return s/(n1*n2);
}

int main(){
    tqdm bar;
    //std::cout << "test";
    int num_inp = 3;
    int num_h = 10;
    int num_h_out = 20; 
    int num_osc = 4;
    int num_out = 4;
    float dt = 0.001;
    float lr = 0.001;

    int *Tsw;
    int *Tst;
    int *theta;
    Tsw = new int[5];
    Tst = new int[5];
    theta = new int[5];
    Tsw[0] = 20;
    Tst[0] = 60;
    theta[0] = 30;
    Tsw[1] = 30; 
    Tst[1] = 90; 
    theta[1] = 30;
    Tsw[2] = 10; 
    Tst[2] = 30; 
    theta[2] = 30;
    Tsw[3] = 40; 
    Tst[3] = 120; 
    theta[3] = 30;
    Tsw[4] = 10;
    Tst[4] = 70;
    theta[4] = 30;    

    //std::cout << "here"; 
    int N = 960;
    //std::cout << "here2";    
    //InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N, lr);   
    DataLoader data(num_osc, num_out, 5, dt, Tsw, Tst, theta, N, 0.6);
    std::complex<float> **Z;
    std::complex<float> **Y;
    float error = new float[nepochs];
    float temp;
    float **signal;
    float *freq;
    int range = new int[nepochs]
    for(int i =0; i<nepochs; i++){
        bar.progress(i, nepochs);
        for(int j = 0; j<4; j++){
            //inp.forwardPropagation(input[j], Y_inp_mlp);
            signal = data.getSignal(j);
            freq = data.getInput(j);
            Z = osc.forwardPropagation(freq);
            Y = out.forwardPropagation(Z);
            temp +=mse(signal, Y, num_out, N);
            out.backwardPropagation(signal, Z);
        }
        error[i] = temp/4;
        range[i] = i;
    }
    float *time;
    float *y;
    time = new float[N];
    y = new float[N];
    signal = data.getSignal(4);
    freq = data.getInput(4);
    Z = osc.forwardPropagation(freq);
    Y = out.forwardPropagation(Z);
    for(int i=0; i<N; i++){
        time[i] = i*dt;
    }
    bar.finish();
    plt::suptitle("Training Plot");
    plt::subplot(1, 9, 1);
    plt::plot(, range, error);
    plt::title("Error Plot");
    for(int i=0; i<num_osc-1; i=i+2){
        for(int j=0; j<N; j++){
            y[j] = Y[i][j];
        }
        plt::subplt(1, 9, i+2);
        plt::named_plot("hip" + to_string(i+1) + "signal", time, signal[i]);
        plt::named_plot("hip" + to_string(i+1) + "prediction", time, y );
        for(int j=0; j<N; j++){
            y[j] = Y[i+1][j];
        }
        plt::subplot(1, 9, i+3);
        plt::named_plot("knee" + to_string(i+1) + "signal", time, signal[i+1]);
        plt::named_plot("knee" + to_string(i+2) + "knee", time, y);
    }
    plt::save("../../images/training_plot_output_mlp_exp9.png");
    
    return 0;
}
