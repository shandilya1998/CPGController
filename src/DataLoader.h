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

class DataLoader{
    private:
        float Tsw;
        float Tst;
        float T;
        float beta;
        float *phaseOffset;
        float *A;
        int num_osc;
        int N;
    public:
        DataLoader(int num_osc, int n, float Tsw, float Tst, float *phaseOffset, float *A);
        void HipJoint(float **out);
        void KneeJoint(float **out);
        void InputVector(float *out);
}
