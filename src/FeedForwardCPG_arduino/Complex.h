#ifndef MATH
#define MATH
#include <math.h>
#endif

class Complex{
    private:
        float real;
        float imag;
    public:
        float getReal(){return real;}
        float getImag(){return imag;}
        float setReal(float r){real = r;}
        float setImag(float i){imag = i;}
};
