#include <data.h>
#include <FeedForwardCPG.h>

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


Servo servo0;
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;
Servo servo7;

float freq[10] = {
  78.74, 
  157.49,
  236.24,
  314.98,
  393.73,
  472.48,  
  551.22,  
  629.97,  
  708.72,
  787.47};

float dt = 0.001;
int Tst = 60;
int Tsw = 20;
int theta = 15;
int i;
Complex *y;
float *out = new float[num_out];

FeedForwardCPG net(dt, w1_real, w1_imag, w2_real, w2_imag);
void setup() {
  // put your setup code here, to run once:
  //Serial.begin(9600);
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
  y = net.feedforwardPropagation(freq);
  for(i =0; i<num_out; i++){
    out[i] = theta*y[i].getReal();
    }
  /*
  servo0.write(90+out[0]);
  servo1.write(90+out[1]);
  servo2.write(90+out[2]);
  servo3.write(90+out[3]);
  servo4.write(90+out[4]);
  servo5.write(90+out[5]);
  servo6.write(90+out[6]);
  servo7.write(90+out[7]);
  */
  Serial.println(90);
  Serial.print("\t");
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
  delay(10);
}
