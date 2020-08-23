#include "Activation.h"

//Need to change all 
void Activation::sigmoidf(float *inp, float *out, struct ParamsSigmoid params){
    /*
    */  
    float unity = 1.0;
    for(int i =0; i<params.dim; i++){
        out[i] = -unity + 2*params.upperBound/(
            unity+exp(
                -params.weight*inp[i]
            )
        );
    }
}

void Activation::sigmoidgrad(float *inp,float *out, struct ParamsSigmoid params){
    /*
    */
    for(int i =0; i<params.dim; i++){ 
    }
}

void Activation::sigmoidf(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid params){
    /*
        * float params array of length 2
        * params.dim - input dimensionality
        * params.upperBound - upper bound of sigmoid values 
                    - real constant
                    - 1 <= bound
    */
    std::complex<float> upperBound(params.upperBound, 0);
    float unity = 1.0;
    std::complex<float> iota(0, 1);
    for(int i =0; i<params.dim; i++){
        out[i] = params.upperBound*(
            unity/(
                unity+exp(
                    -params.weight*inp[i].real()
                )
            )
        )+iota*params.upperBound*(
            unity/(
                unity+exp(
                    -params.weight*inp[i].imag()
                )
            )
        );
    }
}

void Activation::sigmoidgrad(std::complex<float> *inp, std::complex<float> *out, struct ParamsSigmoid params){
    /*
        * float params array of length 2
        * params[0] - input dimensionality
        * params[1] - upper bound of sigmoid values 
                    - bound != 0
    */
    float exponent = 2.0;
    float unity = 1.0;
    float n_unity = -1.0;
    std::complex<float> iota(0, 1);
    for(int i =0; i<params.dim; i++){
        out[i] = params.upperBound*(
            n_unity*exp(
                n_unity*inp[i].real()
            )/pow(
                unity+exp(
                    n_unity*inp[i].real()
                ),
                exponent
            )
        )+iota*params.upperBound*(
            n_unity*exp(
                n_unity*inp[i].imag()
            )/pow(
                unity + exp(
                    n_unity*inp[i].imag()
                ),
                exponent
            )   
        ); 
    }
}

