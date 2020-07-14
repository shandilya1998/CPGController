#include "InputMLP.h"
#include "OscLayer.h"
#include "OutputMLP.h"
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
    int num_osc = 3;
    int num_out = 3;
    float dt = 0.001;
    //std::cout << "here"; 
    int N = static_cast<int>(3*M_PI/dt);
    //std::cout << "here2";    
    InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N);   
    return 0;
}
