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
        float *A_h;
        float *A_k;
        int num_osc;
        int N;
    public:
        DataLoader(int num_osc, int n, float Tsw, float Tst, float *phaseOffset, float *a_h, float *a_k);
        void HipJoint(float **out);
        void KneeJoint(float **out);
        void InputVector(float *out);
}
