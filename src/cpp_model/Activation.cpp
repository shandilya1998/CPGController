#include "Activation.h"

//Need to change all 
void Activation::reluf(float *inp, float *out, struct ParamsRelu *params){
    /*
        * float params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params->dim;i++){
        if(inp[i]<=0){
            out[i] = params->weight*inp[i];
        }
    }     
}

void Activation::relugrad(float *inp, float *out, struct ParamsRelu *params){
    /*  
        * float params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    for(int i=0; i<params->dim;i++){
        if(inp[i]<=0){
            out[i] = params->weight;
        }   
        else{
            out[i] = 1.0;
        }   
    }    
}

void Activation::reluf(std::complex<float> *inp, std::complex<float> *out, struct ParamsRelu *params){
    /*
        * struct with two child elements 
        * params.dim - input dimensionality
        * params.weight - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    std::complex<float> weight(params->weight, 0);
    for(int i=0; i<params->dim;i++){
        if(real(inp[i])<=0 || imag(inp[i])<=0){
            out[i] = weight*inp[i];
        }
    }
}

void Activation::relugrad(std::complex<float> *inp, std::complex<float> *out, struct ParamsRelu *params){
    /*  
        * float params array of length 2 
        * params[0] - input dimensionality
        * params[1] - weight for inputs less that 0
                    - 0 <= weight < 1
    */
    std::complex<float> weight(params->weight, 0);
    for(int i=0; i<params->dim;i++){
        if(real(inp[i])<=0 || imag(inp[i])<=0){
            out[i] = weight;
        }   
        else{
            out[i] = 1.0;
        }   
    }    
}

void Activation::sigmoidf(float *inp, float *out, struct ParamsSigmoid *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - 1 <= bound
    */  
    for(int i =0; i<params->dim; i++){
        out[i] = params->upperBound/(
            1+exp(
                -params->weight*inp[i]
            )
        );
    }
}

void Activation::sigmoidgrad(float *inp,float *out, struct ParamsSigmoid *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        out[i] = params->upperBound*param->weight*exp(
            -params->weight*inp[i]
        )/pow(
            1+exp(
                -params->weight*inp[i]
            ), 
            2.0);
    }
}

void Activation::sigmoidf(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid *params){
    /*
        * float params array of length 2
        * params.dim - input dimensionality
        * params.upperBound - upper bound of sigmoid values 
                    - real constant
                    - 1 <= bound
    */
    std::complex<float> upperBound(params->upperBound, 0);
    float c1 = 1.0;
    std::complex<float> iota(0, 1);
    for(int i =0; i<params->dim; i++){
        out[i] = params->upperBound*(
            1/(
                1+exp(
                    -params->weight*real(
                        inp[i]
                    )
                )
            )
        )+iota*params->upperBound*(
            1/(
                1+exp(
                    -params->weight*imag(
                        inp[i]
                    )
                )
            )
        );
    }
}

void Activation::sigmoidgrad(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    std::complex<float> c1(1, 0);
    std::complex<float> iota(0, 1);
    for(int i =0; i<params->dim; i++){
        out[i] = params->upperBound*(
            -exp(
                -real(
                    inp[i]
                )
            )/pow(
                (
                    1+exp(
                        -real(
                            inp[i]
                        )
                    ),
                    2
                )
            )
        )+iota*params->upperBound*(
                -exp(
                    -image(
                        inp[i]
                    )
                )/pow(
                    (1+exp(
                        -imag(
                            inp[i]
                        )
                    ),
                    2
                )   
            )
        ) 
    }
}

void Activation::tanhf(float *inp, float *out, struct ParamsTanh *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        out[i] = params->bound*tanh(inp[i]);
    }
}

void Activation::tanhgrad(float *inp, float *out, struct ParamsTanh *params){
    /*  
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        out[i] = params->bound*(1-pow(tanh(inp[i]), 2.0));
    }   
}

void Activation::tanhf(std::complex<float> *inp, std::complex<float> *out, struct ParamsTanh *params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    std::complex<float> iota(0, 1);
    for(int i =0; i<params->dim; i++){
        out[i] = params->bound*(tanh(real(inp[i]))+iota*imag(inp[i]));
    }
}

void Activation::tanhgrad(std::complex<float> *inp, std::complex<float> *out, struct ParamsTanh *params){
    /*  
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - bound of tanh values 
                    - bound != 0
    */
    for(int i =0; i<params->dim; i++){
        out[i] = params->bound*(pow(cosh(real(inp[i])), -2.0))+params->iota*bound*(pow(cosh(imag(inp[i])), -2.0));
    }
}
