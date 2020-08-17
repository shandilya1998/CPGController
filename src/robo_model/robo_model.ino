#include <data.h>

#include "Arduino.h"
#include <Servo.h>
#include <math.h>

#define SERVO0_PIN 2
#define SERVO1_PIN 3
#define SERVO2_PIN 4
#define SERVO3_PIN 5
#define SERVO4_PIN 6
#define SERVO5_PIN 7
#define SERVO6_PIN 8
#define SERVO7_PIN 9
#define LPERIOD 10000L  // loop period time in us.
#define T LPERIOD/1000000.0 // T=10ms.
#define num_osc 8
#define num_out 8
#define num_h 16

int i, j;

class Complex{
  private:
    float real;
    float imag;
  public:
    Complex(float r = 0, float i = 0){
      real = r;
      imag = i;
      }
    float getReal(){return real;}
    float getImag(){return imag;}
    void setReal(float r){real = r;}
    void setImag(float i){imag = i;}
  };

class FeedForwardCPG{
  private:
    float dt;
    float *r;
    float *phi;
  public:
    float *y;
    FeedForwardCPG(float DT);
    float* forwardPropagation(float *omega);
  };

FeedForwardCPG::FeedForwardCPG(float DT){
  dt = DT;
  y = new float[num_out];
  r = new float[num_osc];
  phi = new float[num_osc];
  for(i =0; i<num_osc; i++){
    r[i] = 1.0;
    phi[i] = 0.0;
    }
  //Serial.begin(9600);
  }

float* FeedForwardCPG::forwardPropagation(float *omega){
  Complex *Z = new Complex[num_osc];
  Complex *X1 = new Complex[num_h];
  float temp_r, temp_i;
  for(i=0; i<num_osc; i++){
    r[i] = r[i] + (1-pow(r[i], 2)*r[i]*dt);
    phi[i] = phi[i] + omega[i]*dt;
    Z[i].setReal(r[i]*cos(phi[i]));
    //Serial.println("real");
    //Serial.println(Z[i].getReal());
    //Serial.print("\n");
    Z[i].setImag(r[i]*sin(phi[i]));
    //Serial.println("imag");
    //Serial.println(Z[i].getImag());
    //Serial.print("\n");
    }
  for(i=0; i<num_h; i++){
    temp_r=0.0;
    temp_i=0.0;
    for(j=0; j<num_osc; j++){
        temp_r += w1_real[i][j]*Z[j].getReal()-w1_imag[i][j]*Z[j].getImag();
        temp_i += w1_real[i][j]*Z[j].getImag()+w1_imag[i][j]*Z[j].getReal();
      }
      X1[i].setReal(1/(1+exp(-0.5*temp_r)));
      //Serial.println("real");
      //Serial.println(X1[i].getReal());
      X1[i].setImag(1/(1+exp(-0.5*temp_i))); 
      //Serial.println("imag");
      //Serial.println(X1[i].getImag());
    }
  for(i=0; i<num_out; i++){
    temp_r = 0.0;
    for(j=0; j<num_osc; j++){
        temp_r += w2_real[i][j]*Z[j].getReal()-w2_imag[i][j]*Z[j].getImag();
      }
      y[i] = 1/(1+exp(-0.5*temp_r));
      //Serial.println(y[i]);
    }
  delete Z;
  delete X1;
  }

Servo servo0;
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;
Servo servo7;

float freq[8] = {
  78.74, 
  157.49,
  236.24,
  314.98,
  393.73,
  472.48,  
  551.22,  
  629.97};

float dt = 0.001;
int Tst = 60;
int Tsw = 20;
int theta = 45;
Complex *y;
float *out = new float[num_out];

FeedForwardCPG net(dt);
void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
  Serial.print("Hello");
  servo0.attach(SERVO0_PIN);
  servo1.attach(SERVO1_PIN);
  servo2.attach(SERVO2_PIN);
  servo3.attach(SERVO3_PIN);
  servo4.attach(SERVO4_PIN);
  servo5.attach(SERVO5_PIN);
  servo6.attach(SERVO6_PIN);
  servo7.attach(SERVO7_PIN);
}

void loop() {
  net.forwardPropagation(freq);
  for(i =0; i<num_out; i++){
    out[i] = theta*net.y[i];
    }
  
  servo0.write(90+out[0]);
  servo1.write(75+out[1]);
  servo2.write(90+out[2]);
  servo3.write(75+out[3]);
  servo4.write(90+out[4]);
  servo5.write(75+out[5]);
  servo6.write(90+out[6]);
  servo7.write(75+out[7]);
  //*/
  /*
  Serial.println(out[0]);
  Serial.print("\t");
  Serial.println(out[1]);
  Serial.print("\t");
  Serial.println(out[2]);
  Serial.print("\t");
  Serial.println(out[3]);
  Serial.print("\t");
  Serial.println(out[4]);
  Serial.print("\t");
  Serial.println(out[5]);
  Serial.print("\t");
  Serial.println(out[6]);
  Serial.print("\t");
  Serial.println(out[7]); 
  //delay(10);
  //*/
  delay(100);
}
