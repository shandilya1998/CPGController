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

int main(){
    tqdm bar;
    //std::cout << "test";
    int num_inp = 3;
    int num_h = 10;
    int num_h_out = 20; 
    int num_osc = 4;
    int num_out = 4;
    double dt = 0.001;
    int Tsw = 5;
    int Tst = 15;
    double offset[4] = {0.0, 0.0, 0.0, 0.0};
    double A_h = 15.0;
    double A_k = 15.0;
    //std::cout << "here"; 
    int N = static_cast<int>(3*M_PI/dt);
    //std::cout << "here2";    
    //InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N);   
    DataLoader data(num_osc, N, Tsw, Tst, offset, A_h, A_k);
    double **input = new double *[4];
    for(int i =0; i<4; i++){
        input[i] = new double[3];
    }
    double ***output = new double **[4];
    for(int i = 0; i<4; i++ ){
        output[i] = new double *[2*num_out];
        for(int j = 0; j<2*num_out; j++){
            output[i][j] = new double [N];
        }
    } 
    data.getModelInput(input[0]);
    data.getModelOutput(output[0]);
    data.setTsw(3);
    data.getModelInput(input[1]);
    data.getModelOutput(output[1]);
    data.setTsw(5);
    data.setTst(25);
    data.getModelInput(input[2]);
    data.getModelOutput(output[2]);
    data.setTsw(3);
    data.getModelInput(input[3]);
    data.getModelOutput(output[3]);
    int nepochs = 3000;
    double *Y_inp_mlp = new double[num_osc];
    double **phase = new double *[num_osc];
    std::complex<double> **Z_out_osc = new std::complex<double> *[N];
    for(int i=0; i<num_osc; i++){
        phase[i] = new double[N];
        Z_out_osc[i] = new std::complex<double>[N];
    }
    double **y = new double *[num_out];
    for(int i = 0; i<num_out; i++){
        y[i] = new double[N];    
    }
    double **W1_inp_mlp, **W2_inp_mlp, *X_h_inp_mlp;
    std::complex<double> **W1_out_mlp, **W2_out_mlp, **X_h_out_mlp;
    out.getInputWeights(W1_out_mlp);
    out.getHiddenWeights(W2_out_mlp);     
    out.getXh(X_h_out_mlp);
    inp.getInputWeights(W1_inp_mlp);
    inp.getHiddenWeights(W2_inp_mlp);    
    inp.getXh(X_h_inp_mlp);
    for(int i =0; i<nepochs; i++){
        bar.progress(i, nepochs);
        for(int j = 0; j<4; j++){
            //inp.forwardPropagation(input[j], Y_inp_mlp);
            osc.forwardPropagation(Y_inp_mlp, Z_out_osc, phase);
            out.forwardPropagation(Z_out_osc, y);
        }
    }    
    bar.finish();
    return 0;
}
