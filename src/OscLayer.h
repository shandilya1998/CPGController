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
        float *omega;
        std::complex<float> *Z; 
        float *phi;
        int num_osc;
        float dt;
        int N;
    public:
        OscLayer(int n_o, int n, float dT);
        void computeZ(float *freq, std::complex<float> *out);
        void forwardPropagation(float *omega, std::complex<float> **Zout, float **phaseOut);
};
