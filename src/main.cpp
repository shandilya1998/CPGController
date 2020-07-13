#include "InputMLP.h"
#include "OscLayer.h"
#include "OutputMLP.h"
#ifndef COMPLEX
#define COMPLEX

#include <complex>

#endif

int main(){
    int num_inp = 3;
    int num_h = 10;
    int num_h_out = 20; 
    int num_osc = 3;
    int num_out = 3;
    float dt = 0.001;
    int N = static_cast<int>(3*M_PI/dt);
    float W1[num_h][num_inp];
    float W2[num_osc][num_h];
    std::complex<float> W3[num_osc][num_h_out];
    std::complex<float> W4[num_h_out][num_out];
    InputMLP inp(num_inp, num_h, num_osc);
    OscLayer osc(num_osc, N, dt);
    OutputMLP out(num_osc, num_h_out, num_out, N);   
    return 0;
}
