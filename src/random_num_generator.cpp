#include "random_num_generator.h"

float get_random()
{
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1
    return dis(e);
}

std::complex<float> get_complex_random(){
    static std::default_random_engine e;
    static std::uniform_real_distribution<> dis(0, 1); // range 0 - 1
    std:complex<float> rand_num = (dis(e), dis(e));
    return rand_num;
}
