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
        double *omega;
        int num_osc;
        double dt;
        int N;
    public:
        OscLayer(int n_o, int n, double dT);
        void computeZ(double freq, std::complex<double> *out, double *phi);
        void forwardPropagation(double *omega, std::complex<double> **Zout, double **phaseOut);
};
