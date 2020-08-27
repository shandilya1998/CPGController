//#include "InputMLP.h"
#define WITHOUT_NUMPY
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
#include <math.h>
#include "random_num_generator.h"
#endif
#ifndef TQDM
#define TQDM
#include "tqdm.h"
#endif
#include "matplotlibcpp.h"
#include <string>
#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

namespace plt = matplotlibcpp;

float mse(float **Y, std::complex<float> **Y_p, int n1, int n2){
    float s=0;
    for(int i=0;i<n1;i++){
        for(int j=0; j<n2;j++){
            s += pow(Y[i][j]-Y_p[i][j].real(), 2);
        }
    }
    return s/(n1*n2);
}

int main(){
    //std::cout << "test";
    int num_inp = 3;
    int num_h = 10;
    int num_h_out = 16; 
    int num_osc = 8;
    int num_out = 8;
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
    int nepochs = 100;
    //std::cout << "here"; 
    int N = 1440;
    //std::cout << "here2";    
    //InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N, lr);   
    DataLoader data(num_osc, num_out, 5, dt, Tsw, Tst, theta, N, 0.6);
    //std::vector<float> vec(N);
    data.setup();
    std::complex<float> **Z;
    std::complex<float> **Y;
    std::vector<float> error(nepochs);
    float temp;
    float **signal;
    float *freq;
    std::vector<float> range(nepochs);
    tqdm bar; 
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
        error.at(i) = temp/4;
        if(i%10==0){
            std::cout<<"error:"<<error[i]<<"\n";
        }
        range.at(i) = i;
    }
    std::vector<float> time(N);
    std::vector<float> sig(N);
    std::vector<float> y(N);
    std::vector<float> x(N);
    signal = data.getSignal(4);
    freq = data.getInput(4);
    Z = osc.forwardPropagation(freq);
    Y = out.forwardPropagation(Z);
    for(int i=0; i<N; i++){
        time.at(i) = i;
    }
    bar.finish();
    plt::figure();
    plt::figure_size(720 ,6490);
    plt::subplot(9, 1, 1);
    plt::named_plot("Training Error", range, error);
    plt::legend();
    plt::title("Error Plot");
    for(int i=0; i<num_osc; i=i+2){
        for(int j=0; j<N; j++){
            y[j] = Y[i][j].real();
            x[j] = signal[i][j];
        }
        plt::subplot(9, 1, i+2);
        plt::named_plot("hip " + std::to_string(i+1) + " signal", time, x, "r");
        plt::named_plot("hip " + std::to_string(i+1) + " prediction", time, y, "b" );
        plt::legend();
        plt::title("Hip " + std::to_string(i/2+1));
        for(int j=0; j<N; j++){
            y.at(j) = Y[i+1][j].real();
            x.at(j) = signal[i+1][j];
        }
        plt::subplot(9, 1, i+3);
        plt::named_plot("knee " + std::to_string(i+1) + " signal", time, x, "r");
        plt::named_plot("knee " + std::to_string(i+2) + " prediction", time, y, "b");
        plt::legend();
        plt::title("Knee " + std::to_string(i/2+1));
    }
    plt::show();
    plt::save("training_plot_output_mlp_exp9.png");
    return 0;
}
