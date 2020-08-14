#include "Arduino.h"
#include <Servo.h>
#include <math.h>

//setting pin number for hip servos
#define SERVO_PIN0 2
#define SERVO_PIN1 3
#define SERVO_PIN2 4
#define SERVO_PIN3 5
//setting pin number for knee servos
#define SERVO_PIN4 6
#define SERVO_PIN5 7
#define SERVO_PIN6 8
#define SERVO_PIN7 9
#define LPERIOD 10000L


unsigned long nextLoop;
Servo servo0;
Servo servo1;
Servo servo2;
Servo servo3;
Servo servo4;
Servo servo5;
Servo servo6;
Servo servo7;
int Tsw = 20;
int Tst = 60;
int theta = 15;
int T;
int t = 0; int i;
float beta;
void setup() {
  // put your setup code here, to run once:
  T = Tsw+Tst;
  beta= Tst/T;
  servo0.attach(SERVO_PIN0);
  servo1.attach(SERVO_PIN1);
  servo2.attach(SERVO_PIN2);
  servo3.attach(SERVO_PIN3);
  servo4.attach(SERVO_PIN4);
  servo5.attach(SERVO_PIN5);
  servo6.attach(SERVO_PIN6);
  servo7.attach(SERVO_PIN7);
  nextLoop=micros()+LPERIOD;
}

void loop() {
  // put your main code here, to run repeatedly:
  for(i=0;i<=T;i++){
    t = i+T;
    if(i>=0 && i<T*beta/2){
      servo0.write(90+theta*sin(M_PI*t/(T*beta)+M_PI));
      delay(5);
      servo2.write(90+theta*sin(M_PI*(t-T/4)/(T*beta)+M_PI));
      delay(5);
      servo4.write(90+theta*sin(M_PI*(t-T/2)/(T*beta)+M_PI));
      delay(5);
      servo6.write(90+theta*sin(M_PI*(t-3*T/4)/(T*beta)+M_PI));
      }
    else if(i>=T*beta/2 && i<T*(2-beta)/2){
      servo0.write(90+theta*sin(M_PI*t/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta))));
      delay(5);
      //servo1.write(90+theta*sin(M_PI*t/(T*(1-beta))-M_PI*(beta-1)/beta));
      delay(5);
      servo2.write(90+theta*sin(M_PI*(t-T/4)/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta))));
      delay(5);
      //servo3.write(90+theta*sin(M_PI*(t-T/4)/(T*(1-beta))-M_PI*(beta-1)/beta));
      delay(5);
      servo4.write(90+theta*sin(M_PI*(t-T/2)/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta))));
      delay(5);
      //servo5.write(90+theta*sin(M_PI*(t-T/2)/(T*(1-beta))-M_PI*(beta-1)/beta));
      delay(5);
      servo6.write(90+theta*sin(M_PI*(t-3*T/4)/(T*(1-beta))+M_PI*(3-4*beta)/(2*(1-beta))));
      delay(5);
      //servo7.write(90+theta*sin(M_PI*(t-3*T/4)/(T*(1-beta))-M_PI*(beta-1)/beta));
      }
    else{
      servo0.write(90+theta*sin(M_PI*t/(T*beta)+M_PI*(beta-1)/beta));
      delay(5);
      servo2.write(90+theta*sin(M_PI*(t-T/4)/(T*beta)+M_PI*(beta-1)/beta));
      delay(5);
      servo4.write(90+theta*sin(M_PI*(t-T/2)/(T*beta)+M_PI*(beta-1)/beta));
      delay(5);
      servo6.write(90+theta*sin(M_PI*(t-3*T/4)/(T*beta)+M_PI*(beta-1)/beta));
      }
     delay(10);
  }
}
