#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif

#ifndef PYIN
#define PYIN
#include "LibPyin/source/libpyincpp.h"
#endif

#ifndef TQDM
#define TQDM
#include "tqdm.h"
#endif

class DataLoader{
    private:
        int num_osc;
        int num_out;
        int num_d;
        int *Tsw;
        int *Tst;
        int *theta;
        float dt;
        float *ff;
        float ***signal;
        int N;
        float *beta;
        void _populate(float **out, int index);
        float **base;
    public:
        DataLoader(
            int n_o, 
            int n_out, 
            int num, 
            float DT, 
            int *tsw, 
            int *tst,
            int *t,
            int n
        );
        float* getFundamentFrequencies(){
            return ff
        }  
        void createSignals();
};
