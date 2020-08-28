#pragma once

#ifndef COMMON_STD_IMPORTS
#define COMMON_STD_IMPORTS
#include <iostream>
#include <math.h>
#include "random_num_generator.h"
#endif

class CFourier
{
public:
	double pi;
	unsigned long int fundamental_frequency;
	unsigned int sample_rate;
    unsigned long number_of_samples;
    float *vector;
    CFourier(void);
	~CFourier(void);
	// FFT 1D
	float ComplexFFT(float data[], int sign);
    void build(unsigned int sr, unsigned long ns);
};

