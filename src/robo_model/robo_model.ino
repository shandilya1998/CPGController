#include <random_num_generator.h>
#include <OutputMLP.h>
#include <InputMLP.h>
#include <DataLoader.h>
#include <OscLayer.h>
#include <Activation.h>
#include <tqdm.h>

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
#define N 4 // number of oscillators
#define T LPERIOD/1000000.0 // T=10ms.

Servo servo0;
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;
Servo servo7;

float freq[20] = {
  78.74680994, 
  157.49361989,
  236.24042983,
  314.98723978,
  393.73404972,
  472.48085967,  
  551.22766961,  
  629.97447956,  
  708.7212895,
  787.46809945,
  866.21490939,  
  944.96171934, 
  1023.70852928,
  1102.45533923, 
  1181.20214917,
  1259.94895912, 
  1338.69576906, 
  1417.44257901, 
  1496.18938895, 
  1574.9361989};

int num_osc = 20;
int num_h = 50;
int num_out = 8;
float = dt = 0.001;
int N = 500;
int Tst = 60;
int Tsw = 20;
int theta = 15;

OscLayer osc(num_osc, N, dt);
OutputMLP out(num_osc, num_h, num_out, N);

void setup() {
  // put your setup code here, to run once:
  Serial.begin(9600);
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
  // put your main code here, to run repeatedly:
  
}
