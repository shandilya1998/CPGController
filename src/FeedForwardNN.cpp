#include "InputMLP.h"
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

int main(){
    //std::cout << "test";
    int num_inp = 3;
    int num_h = 10;
    int num_h_out = 20; 
    int num_osc = 4;
    int num_out = 4;
    float dt = 0.001;
    int Tsw = 5;
    int Tst = 15;
    float offset[4] = {0.0, 0.0, 0.0, 0.0};
    float A_h = 15.0;
    float A_k = 15.0;
    //std::cout << "here"; 
    int N = static_cast<int>(3*M_PI/dt);
    //std::cout << "here2";    
    InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N);   
    DataLoader data(num_osc, N, Tsw, Tst, offset, A_h, A_k);
    float **input = new float *[4];
    for(int i =0; i<4; i++){
        input[i] = new float[3];
    }
    float ***output = new float **[4];
    for(int i = 0; i<4; i++ ){
        output[i] = new float *[2*num_osc];
        for(int j = 0; j<2*num_osc; j++){
            output[i][j] = new float [N];
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
    for(int i =0; i<nepochs; i++){
        for(int j = 0; j<4; j++){
            
        }
    }    
    return 0;
}
