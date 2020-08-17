/* Sweep
 by BARRAGAN <http://barraganstudio.com>
 This example code is in the public domain.

 modified 8 Nov 2013
 by Scott Fitzgerald
 http://www.arduino.cc/en/Tutorial/Sweep
*/

#include <Servo.h>
int servo0 = 2;
int servo1 = 3;
int servo2 = 4;
int servo3 = 5;
int servo4 = 6;
int servo5 = 7;
int servo6 = 8;
int servo7 = 9;

Servo myservo0;
Servo myservo1;
Servo myservo2;
Servo myservo3;
Servo myservo4;
Servo myservo5;
Servo myservo6;
Servo myservo7;
// create servo object to control a servo
// twelve servo objects can be created on most boards

int pos1 = 90; 
int pos2 = 60;// variable to store the servo position

void setup() {
  Serial.begin(9600);
  Serial.print("Hello");
  myservo0.attach(servo0);
  myservo1.attach(servo1);
  myservo2.attach(servo2);
  myservo3.attach(servo3);
  myservo4.attach(servo4);
  myservo5.attach(servo5);
  myservo6.attach(servo6);
  myservo7.attach(servo7);
}

void loop() {                     
  myservo0.write(pos1);
  myservo1.write(pos2);
  myservo2.write(pos1);
  myservo3.write(pos2);
  myservo4.write(pos1);
  myservo5.write(pos2);
  myservo6.write(pos1);
  myservo7.write(pos2);
}
