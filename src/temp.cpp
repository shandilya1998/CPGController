#include "random_num_generator.h"
#include <iostream>

void outrandom(float *o){
    for(int i = 0; i<10; i++){
        o[i] = get_random();
    }
}

int main(){
    float *temp = new float[5];
    float **out = new float *[4];
    /*
    for(int i = 0; i< 4; i++){
        out[i] = new float[5];
        outrandom(out[i]);
    }
    for(int i = 0; i<4;i++){
        for(int j = 0; j< 5; j++){
            std::cout << out[i][j] << "\t";
        }
    }   std::cout << "\n";
    */
    int N = 10;
    for(int i=0,j=0;i<N && j<N;i++,j++){
        std::cout << i << "\t";
        std::cout << j << std::endl;
        
    } 
    return 0;
}
