#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <cmath>
#include "random_num_generator.h"
#endif

#ifndef VECTOR
#define VECTOR
#include <vector>
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
        float ***signals;
        float *input;
        int N;
        float *beta;
        void _populate(float **out, int index);
        float **base;
        void createSignals();
        void calcFF();
        PyinCpp **pyin;           
    public:
        DataLoader(
            int n_o, 
            int n_out, 
            int num, 
            float DT, 
            int *tsw, 
            int *tst,
            int *t,
            int n,
            float cutoff
        );
        ~DataLoader(){
            delete Tsw;
            delete Tst;
            delete theta;
            delete ff;
            delete signals;
            delete input;
            delete beta;
            delete base;
            delete pyin;
        }
        float* getInput(int index){
            for(int i=0; i<num_osc; i++){
                input[i] = ff[index]*(i+1)*M_PI*2;
            }
            return input;
        }
        float** getSignal(int index){
            return signals[index];

        }
        void setup();
};
