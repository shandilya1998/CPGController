#include <FeedForwardCPG.h>

FeedForwardCPG::FeedForwardCPG(
    float dT, 
    float w1_real[num_h][num_osc], 
    float w1_imag[num_h][num_osc], 
    float w2_real[num_out][num_h], 
    float w2_imag[num_out][num_h]
){
    int i, j;
    dt = dT;
    W1 = new Complex *[num_h];
    for(i=0;i<num_h;i++){
        W1[i]= new Complex[num_osc];
        for(j=0; j<num_osc; j++){
            W1[i][j].setReal(w1_real[i][j]);
            W1[i][j].setImag(w1_imag[i][j]);
        }
    }
    W2 = new Complex *[num_out];
    for(i=0; i<num_out; i++){
        W2[i]= new Complex[num_h];
        for(j=0; j<num_h; j++){
            W2[i][j].setReal(w2_real[i][j]);
            W2[i][j].setImag(w2_imag[i][j]);
        }
    }
    r = new float[num_osc];
    for(i=0; i<num_osc; i++){
        r[i] = 1.0;
    }
    phi = new float[num_osc];
}

Complex* FeedForwardCPG::feedforwardPropagation(float omega[num_out]){
    Complex *Y = new Complex[num_out];
    Complex *X1 = new Complex[num_h];
    Complex *Z = new Complex[num_osc];
    for(int i=0; i<num_osc; i++){
        r[i] = r[i] + (1-pow(r[i], 2))*r[i]*dt;
        phi[i] = phi[i] + omega[i]*dt;
        Z[i].setReal(r[i]*cos(phi[i]));
        Z[i].setImag(r[i]*sin(phi[i]));
    } 
    for(int i=0; i<num_h; i++){
        for(int j=0; j<num_osc; j++){
            X1[i] = op.add(X1[i], op.multiply(W1[i][j], Z[j])); 
        }
    }

    for(int i=0; i<num_out; i++){
        for(int j=0; j<num_h; j++){
            Y[i] = op.add(Y[i], op.multiply(W2[i][j], X1[j]));
        }       
    } 
    return Y;
}
