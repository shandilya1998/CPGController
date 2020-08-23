#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS

#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif

#ifndef COMPLEX
#define COMPLEX

#include <complex>

#endif

class OscLayer{
    private:
        int num_osc;
        float dt;
        int N;
        float *r;
        float *phi;
        std::complex<float> **Z;
    public:
        ~OscLayer(){
            delete r;
            delete phi;
            delete Z;
        }
        OscLayer(int n_o, int n, float dT);
        std::complex<float>** forwardPropagation(float *omega);
};
