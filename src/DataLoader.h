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
        float *offset;
        float *A_h;
        float *A_k;
        int num_osc;
        float *heading;
        int N;
        void HipJoint(float *out, int i, int off);
        void KneeJoint(float *out, int i, int off); 
    public:
        DataLoader(int num_osc, int n, float Tsw, float Tst, float *phaseOffset, float *a_h, float *a_k);
        void setHeading(float x, float i);
        void getData(float **out);
}
