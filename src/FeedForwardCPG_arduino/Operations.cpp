#include <Operations.h>

Complex Operations::multiply(Complex x, Complex y){
    Complex res;
    res.setReal(x.getReal()*y.getReal() - x.getImag()*y.getImag());
    res.setImag(x.getReal()*y.getImag() + x.getImag()*y.getReal());
    return res;
}

Complex Operations::add(Complex x, Complex y){
    Complex res;
    res.setReal(x.getReal() + y.getReal());
    res.setImag(x.getImag() + y.getImag());
    return res;
}

Complex Operations::subtract(Complex x, Complex y){ 
    Complex res;
    res.setReal(x.getReal() - y.getReal());
    res.setImag(x.getImag() - y.getImag());
    return res;
}

Complex Operations::divide(Complex x, Complex y){ 
    Complex res;
    res.setReal(x.getReal() + y.getReal());
    res.setImag(x.getImag() + y.getImag());
    return res;
}
