#pragma once

#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <math.h>
#include "random_num_generator.h"
#endif

#include <stdlib.h>

class CFourier
{
private:
	float pi;
	float fs;;
    unsigned long number_of_samples;
    float *vector;
    // FFT 1D
    void fft(float data[]);
    float fftMagnitude(unsigned long k); 
    float fftFrequency(unsigned long k);
public:
    CFourier(void);
	~CFourier(void);
    void build(float FS, unsigned long nn);
    float fundamentalFrequency(float data[]);
};

