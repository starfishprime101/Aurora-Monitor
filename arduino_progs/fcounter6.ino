#include <FreqCounter.h>

void setup() {
  Serial.begin(57600);                    // connect to the serial port
}

long int frq;
void loop() {

 FreqCounter::f_comp= 8;             // Set compensation to 12
 FreqCounter::start(2000);            // Start counting with gatetime of 2000ms
 while (FreqCounter::f_ready == 0)         // wait until counter ready
 
 frq=FreqCounter::f_freq;            // read result
 frq=frq/2.00000;
 Serial.println(frq);                // print result
 delay(20);
}
