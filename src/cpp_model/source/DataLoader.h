#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <math.h>
#include "random_num_generator.h"
#endif

#ifndef VECTOR
#define VECTOR
#include <vector>
#endif

#ifndef PYIN
#define PYIN
#include "pitch_detection.h"
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
        float ***norm_signals;
        float **max;
        float *input;
        int N;
        float *beta;
        void createSignals();
        void calcFF(std::vector<float> &vec);
        std::vector<float> sig;
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
            delete[] Tsw;
            delete[] Tst;
            delete[] theta;
            delete[] input;
            delete[] ff;
            delete[] beta;
            for(int i=0; i<num_d; i++){
                delete[] max[i];
                for(int j=0; j<num_osc; j++){
                    delete[] signals[i][j];
                    delete[] norm_signals[i][j];
                }
                delete[] signals[i];
                delete[] norm_signals[i];
            }
            delete[] max;
            delete[] signals;
            delete[] norm_signals;
            //delete &sig;
        }
        float* getInput(int index){
            for(int i=0; i<num_osc; i++){
                input[i] = ff[index]*(i+1)*M_PI*2;
            }
            return input;
        }
        float** getSignal(int index){
            return norm_signals[index];

        }
        void setup();
};
