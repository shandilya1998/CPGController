#include "random_num_generator.h"

double get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1
    return dis(e);
}

std::complex<double> get_complex_random(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1
    std::complex<double> rand_num = (dis(e), dis(e));
    return rand_num;
}
