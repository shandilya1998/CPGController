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

Servo myservo0;
Servo myservo1;
Servo myservo2;
Servo myservo3;// create servo object to control a servo
// twelve servo objects can be created on most boards

int pos = 90;    // variable to store the servo position

void setup() {
  myservo0.attach(servo0);
  myservo1.attach(servo1);
  myservo2.attach(servo2);
  myservo3.attach(servo3);
}

void loop() {                     
  myservo0.write(pos);
  myservo1.write(pos);
  myservo2.write(pos);
  myservo3.write(pos);
}
