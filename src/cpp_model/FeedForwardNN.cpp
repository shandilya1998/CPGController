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
#include "tqdm.h"
#include "matplotlibcpp.h"

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
    int Tsw = 5;
    int Tst = 15;
    float offset[4] = {0.0, 0.0, 0.0, 0.0};
    float A_h = 15.0;
    float A_k = 15.0;
    //std::cout << "here"; 
    int N = static_cast<int>(3*M_PI/dt);
    //std::cout << "here2";    
    //InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N, lr);   
    DataLoader data(num_osc, N, Tsw, Tst, offset, A_h, A_k);
    std::complex<float> **Z;
    std::complex<float> **Y;
    float error = new float[nepochs];
    float temp;
    float **signal;
    int range = new int[nepochs]
    for(int i =0; i<nepochs; i++){
        bar.progress(i, nepochs);
        for(int j = 0; j<4; j++){
            //inp.forwardPropagation(input[j], Y_inp_mlp);
            signal = data.getSignal(j);
            Z = osc.forwardPropagation(freq[j]);
            Y = out.forwardPropagation(Z);
            temp +=mse(signal, Y, num_out, N);
            out.backwardPropagation(signal, Z);
        }
        error[i] = temp/4;
        range[i] = i;
    }
    bar.finish();
    plt::figure_size(1200, 780);
    plt::plot(range, error);
    plt::save("../../images/training_plot_output_mlp_exp9.png")
    return 0;
}
