import serial
import numpy as np
import time

###*******FGM-3h Sensor Values*********
mSens=0.2744
cSens=11.698
###************************************


currentValue=0.0
#os.chdir('/home/pi/EFM')
#
#
#time.sleep(5)
ser = serial.Serial('/dev/ttyACM0', 57600)
ser.flushInput()
#time.sleep(5)

while True:
    ser.flushInput()
    time.sleep(2)

    
    
    

    while True:
        try:
            currentValue=np.float32(ser.readline())/1000.0       #read value from USB line
            break
        except ValueError:
            print ('failed to read serial line correctly --trying again')
            
    
    print(currentValue, '  kHz')





