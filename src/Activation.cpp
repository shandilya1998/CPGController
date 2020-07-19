#include "Activation.h"

void Activation::reluf(float *inp, float *out, float *params){
    /*
        * float params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params[0];i++){
        if(inp[i]<=0){
            out[i] = params[1]*inp[i];
        }
        else{
            out[i] = inp[i];
        }
    }     
}

void Activation::relugrad(float *inp, float *out, float *params){
    /*  
        * float params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params[0];i++){
        if(inp[i]<=0){
            out[i] = params[1];
        }   
        else{
            out[i] = 1.0;
        }   
    }    
}

void Activation::sigmoidf(float *inp, float *out, float *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - 1 <= bound
    */  
    for(int i =0; i<params[0]; i++){
        out[i] = params[1]/(1+exp(-inp[i]));
    }
}

void Activation::sigmoidgrad(float *inp, float *out, float *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    for(int i =0; i<params[0]; i++){
        out[i] = params[1]*exp(-inp[i])/pow(1+exp(-inp[i]), 2.0);
    }
}

void Activation::tanhf(float *inp, float *out, float *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params[0]; i++){
        out[i] = params[1]*tanh(inp[i]);
    }
}

void Activation::tanhgrad(float *inp, float *out, float *params){
    /*  
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params[0]; i++){
        out[i] = params[1]*(1-pow(tanh(inp[i]), 2.0));
    }   
}
