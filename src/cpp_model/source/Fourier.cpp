#include "Fourier.h"

#define SWAP(a,b) tempr=(a);(a)=(b);(b)=tempr

CFourier::CFourier(void)
{
	pi=4*atan((float)1);
}

CFourier::~CFourier(void)
{
   if(vector!=NULL)
        delete [] vector; 
}

void CFourier::build(float FS, unsigned long ns){ 
    fs = FS;
    number_of_samples = ns;
    vector = new float[2*number_of_samples];
}

// Input: nn is the number of points in the data and in the FFT, 
//           nn must be a power of 2
// Input: data is sampled voltage v(0),0,v(1),0,v(2),...v(nn-1),0 versus time
// Output: vector is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// vector is an array of 2*nn elements
void CFourier::fft(float data[]){
    unsigned long n,mmax,m,j,istep,i;
    float wtemp,wr,wpr,wpi,wi,theta;
    float tempr,tempi;
    for(n=0; n<number_of_samples;n++)
    {   
        vector[2*n]=data[n];
        vector[2*n+1]=0;
    }
	n = number_of_samples<<1;  // n is the size of vector array (2*nn)
	j = 1;
	for(i=1; i<n; i+=2){
		if(j > i){				// bit reversal section
			SWAP(vector[j-1],vector[i-1]);
			SWAP(vector[j],vector[i]);
		}
		m = n>>1;
		while((m >= 2)&&(j > m)){
			j = j-m;
			m = m>>1;
		}
		j = j+m;
	}
	mmax = 2;             // Danielson-Lanczos section
	while( n > mmax){     // executed log2(nn) times
		istep = mmax<<1;
		theta = -6.283185307179586476925286766559/mmax;
		// the above line should be + for inverse FFT
		wtemp = sin(0.5*theta);
		wpr = -2.0*wtemp*wtemp;  // real part
		wpi = sin(theta);        // imaginary part
		wr = 1.0;
		wi = 0.0;
		for(m=1; m<mmax; m+=2){
			for(i=m; i<=n; i=i+istep){
				j = i+mmax;
				tempr     = wr*vector[j-1]-wi*vector[j]; // Danielson-Lanczos formula
				tempi     = wr*vector[j]+wi*vector[j-1];
				vector[j-1] = vector[i-1]-tempr;
				vector[j]   = vector[i]-tempi;
				vector[i-1] = vector[i-1]+tempr;
				vector[i]   = vector[i]+tempi;
			}
			wtemp = wr;
			wr = wr*wpr-wi*wpi+wr;
			wi = wi*wpr+wtemp*wpi+wi;
		}
		mmax = istep;
	}
}

//-----------------------------------------------------------
// Calculates the FFT magnitude at a given frequency index. 
// Input: vector is complex FFT Re[V(0)],Im[V(0)], Re[V(1)],Im[V(1)],...
// Input: nn is the number of points in the data and in the FFT, 
//           nn must be a power of 2
// Input: k is frequency index 0 to nn/2-1
//        E.g., if nn=16384, then k can be 0 to 8191
// Output: Magnitude in volts at this frequency (volts)
// vector is an array of 2*nn elements
// returns 0 if k >= nn/2
float CFourier::fftMagnitude(unsigned long k){
	float nr, realPart, imagPart;

	nr = (float)number_of_samples;
	if (k >= number_of_samples/2){
		return 0.0; // out of range
	}
	if (k == 0){
		return sqrt(vector[0] * vector[0] + vector[1] * vector[1]) / nr;
	}
	realPart = fabs(vector[2*k])   + fabs(vector[2*number_of_samples-2*k]);
	imagPart = fabs(vector[2*k+1]) + fabs(vector[2*number_of_samples-2*k+1]);
	return  sqrt(realPart * realPart + imagPart * imagPart) / nr;
}

//-----------------------------------------------------------
// Calculates equivalent frequency in Hz at a given frequency index. 
// Input: fs is sampling rate in Hz
// Input: nn is the number of points in the data and in the FFT, 
//           nn must be a power of 2
// Input: k is frequency index 0 to nn-1
//        E.g., if nn=16384, then k can be 0 to 16383
// Output: Equivalent frequency in Hz
// returns 0 if k >= nn
float CFourier::fftFrequency (unsigned long k){
	if (k >= number_of_samples){
		return 0.0;     // out of range
	}

	if (k <= number_of_samples/2){
		return fs * (float)k / (float)number_of_samples;
	}
	return -fs * (float)(number_of_samples-k)/ (float)number_of_samples;
}

float CFourier::fundamentalFrequency(float data[]){
    fft(data);
    float max = 0.0, temp, freq;
    for(int i=0; i<number_of_samples; i++){
        temp = fftMagnitude(i);
        if(max<temp){
            max = temp;
            freq = fftFrequency(i);
        }
    }
    return freq*2*M_PI;
}
