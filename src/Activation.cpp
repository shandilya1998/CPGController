#include "Activation.h"

void Activation::reluf(double *inp, struct ParamsRelu *params){
    /*
        * double params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params->dim;i++){
        if(inp[i]<=0){
            inp[i] = params->weight*inp[i];
        }
    }     
}

void Activation::relugrad(double *inp, struct ParamsRelu *params){
    /*  
        * double params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params->dim;i++){
        if(inp[i]<=0){
            inp[i] = params->weight;
        }   
        else{
            inp[i] = 1.0;
        }   
    }    
}

void Activation::reluf(std::complex<double> *inp, struct ParamsRelu *params){
    /*
        * struct with two child elements 
        * params.dim - input dimensionality
        * params.weight - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    std::complex<double> weight(params->weight, 0);
    for(int i=0; i<params->dim;i++){
        if(real(inp[i])<=0 || imag(inp[i])<=0){
            inp[i] = weight*inp[i];
        }
    }
}

void Activation::relugrad(std::complex<double> *inp, struct ParamsRelu *params){
    /*  
        * double params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    std::complex<double> weight(params->weight, 0);
    for(int i=0; i<params->dim;i++){
        if(real(inp[i])<=0 || imag(inp[i])<=0){
            inp[i] = weight;
        }   
        else{
            inp[i] = 1.0;
        }   
    }    
}

void Activation::sigmoidf(double *inp, struct ParamsSigmoid *params){
    /*
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - 1 <= bound
    */  
    for(int i =0; i<params->dim; i++){
        inp[i] = params->upperBound/(1+exp(-inp[i]));
    }
}

void Activation::sigmoidgrad(double *inp, struct ParamsSigmoid *params){
    /*
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        inp[i] = params->upperBound*exp(-inp[i])/pow(1+exp(-inp[i]), 2.0);
    }
}

void Activation::sigmoidf(std::complex<double> *inp, struct ParamsSigmoid *params){
    /*
        * double params array of length 2
        * params.dim - input dimensionality
        * params.upperBound - upper bound of sigmoid values 
                    - real constant
                    - 1 <= bound
    */
    std::complex<double> upperBound(params->upperBound, 0);
    std::complex<double> c1(1, 0);
    for(int i =0; i<params->dim; i++){
        inp[i] = upperBound/(c1+std::exp(-inp[i]));
    }
}

void Activation::sigmoidgrad(std::complex<double> *inp, struct ParamsSigmoid *params){
    /*
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    std::complex<double> upperBound(params->upperBound, 0);
    std::complex<double> c1(1, 0);
    for(int i =0; i<params->dim; i++){
        inp[i] = upperBound*std::exp(-inp[i])/std::pow(c1+std::exp(-inp[i]), 2.0);
    }
}

void Activation::tanhf(double *inp, struct ParamsTanh *params){
    /*
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        inp[i] = params->bound*tanh(inp[i]);
    }
}

void Activation::tanhgrad(double *inp, struct ParamsTanh *params){
    /*  
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        inp[i] = params->bound*(1-pow(tanh(inp[i]), 2.0));
    }   
}

void Activation::tanhf(std::complex<double> *inp, struct ParamsTanh *params){
    /*
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    std::complex<double> bound(params->bound, 0); 
    for(int i =0; i<params->dim; i++){
        inp[i] = bound*std::tanh(inp[i]);
    }
}

void Activation::tanhgrad(std::complex<double> *inp, struct ParamsTanh *params){
    /*  
        * double params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    std::complex<double> bound(params->bound, 0);
    for(int i =0; i<params->dim; i++){
        inp[i] = bound*(std::pow(std::cosh(inp[i]), -2.0));
    }
}
