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
        double beta;
        double *offset;
        double A_h;
        double A_k;
        int num_osc;
        double *heading;
        int N;
        void HipJoint(double *out, int off, double h);
        void KneeJoint(double *out,  int off); 
    public:
        DataLoader(int num_osc, int n, int tsw, int tst, double *o, double a_h, double a_k);
        void getModelInput(double *out);
        void setHeading(double x, double i);
        void getModelOutput(double **out);
        void setTsw(double tsw){ 
            Tsw = tsw;
            T = Tsw+Tst;
            beta = (double) Tst/ (double) T;
        }
        void setTst(double tst){
            Tst = tst;
            T = Tsw+Tst;
            beta = (double) Tst/(double) T;
        }
};
