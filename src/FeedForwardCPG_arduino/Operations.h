#ifndef COMPLEX
#define COMPLEX
#include <Complex.h>
#endif

#ifndef MATH
#define MATH
#include <math.h>
#endif

class Operations{
    public:
        Complex multiply(Complex x, Complex y);
        Complex add(Complex x, Complex y);
        Complex subtract(Complex x, Complex y);
        Complex divide(Complex x, Complex y);
};

