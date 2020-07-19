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
        int Tsw;
        int Tst;
        int T;
        float beta;
        float *offset;
        float A_h;
        float A_k;
        int num_osc;
        float *heading;
        int N;
        void HipJoint(float *out, int off, float h);
        void KneeJoint(float *out,  int off); 
    public:
        DataLoader(int num_osc, int n, int tsw, int tst, float *o, float a_h, float a_k);
        void getModelInput(float *out);
        void setHeading(float x, float i);
        void getModelOutput(float **out);
        void setTsw(float tsw){ 
            Tsw = tsw;
            T = Tsw+Tst;
            beta = (float) Tst/ (float) T;
        }
        void setTst(float tst){
            Tst = tst;
            T = Tsw+Tst;
            beta = (float) Tst/(float) T;
        }
};
