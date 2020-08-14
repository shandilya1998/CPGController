#define num_osc 10
#define num_h 30
#define num_out 8

#ifndef MATH
#define MATH
#include <math.h>
#endif

#ifndef OPERATIONS
#define OPERATIONS
#include <Operations.h>
#endif

class FeedForwardCPG{
    private:
        float dt;
        float *r;
        float *phi;
        Complex **W1;
        Complex **W2;
        Complex *Y;
        Operations op;
    public:
        FeedForwardCPG(){};
        FeedForwardCPG(
            float dT, 
            float w1_real[num_h][num_osc], 
            float w1_imag[num_h][num_osc], 
            float w2_real[num_out][num_h], 
            float w2_imag[num_out][num_h]);
        Complex* feedforwardPropagation(float omega[num_out]);
};
