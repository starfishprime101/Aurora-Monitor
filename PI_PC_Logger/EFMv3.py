#---------------------------ooo0ooo---------------------------
#       EFM monitor v5.0
#       updated 17/Jan/2107
#
#---------------------------Notes---------------------------
#
#   each data point     = 37 bytes
#   at 1 min intervals  = 20Mb p.a
#   dataFile format (txt) = datetime .. B Field nT .. temperature celcius
#
#   activedata stores data for 1 month
#   older data is periodically stripped out and archived
#       to save memory and speed processing
#
#   ActiveData.dat[0][i] = date
#   ActiveData.dat[1][i] = field
#   ActiveData.dat[2][i] = temperature
#
#---------------------------ooo0ooo---------------------------

import serial
import time

import string
import matplotlib
matplotlib.use('Agg') #prevent use of Xwindows
from matplotlib import pyplot as plt
from threading import Thread
import statistics   #used by median filter
import os
import gc

import numpy as np
import time
import array
from array import array
from scipy import signal
from obspy import UTCDateTime, read, Trace, Stream
from shutil import copyfile



#---------------------------ooo0ooo---------------------------
def createMSeed(DataArray, StartDateTime, EndDateTime, nSamples):

  ActualSampleFrequency = float(nSamples) / (EndDateTime - StartDateTime)

  # Fill header attributes
  stats = {'network': 'EM', 'station': '01', 'location': ' ',
         'channel': '1', 'npts': nSamples, 'sampling_rate': ActualSampleFrequency,
         'mseed': {'dataquality': 'D'}}
  # set current time
  stats['starttime'] = StartDateTime
  st = Stream([Trace(data=DataArray[0:nSamples], header=stats)])
  return st

#---------------------------ooo0ooo---------------------------
def SaveDataMSeed(DataArray, StartDateTime, EndDateTime, nDataPoints):

  
  st = createMSeed(DataArray, StartDateTime, EndDateTime, nDataPoints)


  Year = str(StartDateTime.year)
  Month = str(StartDateTime.month)
  Day = str(StartDateTime.day)
  Hour = str(StartDateTime.hour)
  Minute = str(StartDateTime.minute)

  yearDir =  'Data' + '/' + Year

  FileName = str(Year) + '_' + str(Month) + '_' + str(Day) + '__' + Hour +':' + Minute +'.mseed'
  
  here = os.path.dirname(os.path.realpath(__file__))

  try:
    os.makedirs(yearDir)
  except OSError:
    if not os.path.isdir(yearDir):
        raise

  FilePath = os.path.join(here, yearDir, FileName)
  
  dataFile = open(FilePath, 'wb')
  st.write(dataFile,format='MSEED',  encoding=4, reclen=256)
  dataFile.close()
  print ('Data write of active data completed')
  return


#---------------------------ooo0ooo---------------------------
def plotTodaysGraph(DataArray, StartDateTime, EndDateTime, nDataPoints):

  print('plotting DayPlot',nDataPoints)
  
  if (nDataPoints > 20):
    ydata=DataArray[3:nDataPoints]
    

    data=removeGlitches(ydata)
    data=medianFilter(ydata)

    zeroPoint = np.mean(data)
    data=data-zeroPoint #(de-mean the data)
    
    
    yearNow = StartDateTime.year
    monthNow = StartDateTime.month
    dayNow = StartDateTime.day
    
    dateString = str(yearNow) + '-' + str(monthNow) + '-' + str(dayNow)
    filename1 = ('Plots/Today.svg')
    filename2 = 'Plots/' + dateString + '.svg'
    


    YAxisRange=np.amax(data)*1.1
    
    startHour=StartDateTime.hour
    startMinute=StartDateTime.minute
    timeSampled = EndDateTime-StartDateTime  #length of data in seconds
    graphStart = float(startHour) + float(startMinute/60)
    graphEnd = graphStart + (timeSampled/3600.0)
    xValues= np.linspace(graphStart, graphEnd, len(ydata))
    upDated = str(UTCDateTime())
        
    fig = plt.figure(figsize=(12,6))
    xAxisText="Time  updated - "+ upDated +" UTC"
    plt.title(dateString)
    plt.xlabel(xAxisText)
    zeroLabel=str('%0.5g' % (zeroPoint))
    plt.ylabel('$\Delta$Flux Density - nT     0.0='+zeroLabel+'nT')
    plt.plot(xValues, data,  marker='None',    color = 'darkolivegreen')  
    plt.xlim(0, 24.1)
    plt.xticks(np.arange(0, 24.1, 3.0))
    plt.grid(True)
    plt.savefig(filename1)
    
    copyfile('Plots/Today.svg', filename2)   
    plt.close('all')     
    

  return
#---------------------------ooo0ooo---------------------------
def medianFilter(ydata):

  z=len(ydata)
  dataNew=ydata

  for i in range (2,z-2):
    medData=ydata[i]
    dataChunk=[ydata[i-2],ydata[i-1],ydata[i+1],ydata[i+2]]
    medianDatum=statistics.median(dataChunk)
    dataNew[i]=medianDatum
  return dataNew

#---------------------------ooo0ooo---------------------------
def removeGlitches(ydata):
  #to maintain continuity of data-line miss-reads are assigned B=0.0
  z=len(ydata)
  dataNew=ydata

  for i in range (0,z-4):
    if (abs(ydata[i]) < 0.1):
        dataChunk=[ydata[i],ydata[i+1],ydata[i+2],ydata[i+3]]
        dataNew[i]=statistics.median(dataChunk)
  
  dataNew[(z-3):(z-1)]=ydata[(z-3):(z-1)]
  return dataNew
#---------------------------ooo0ooo---------------------------
def runningMeanFast(x, N):
  return np.convolve(x, np.ones((N,))/N)[(N-1):]
#---------------------------ooo0ooo---------------------------
def SaveAndPlot(DataArray, StartDateTime, EndDateTime, nDataPoints):
  SaveDataMSeed(DataArray, StartDateTime, EndDateTime, nDataPoints)
  plotTodaysGraph(DataArray, StartDateTime, EndDateTime, nDataPoints)
  return
#---------------------------ooo0ooo---------------------------


#-------------------------------------------------------------------
#---------------------------ooo0ooo---------------------------
#                                                            #
#                   main body                                #
#                                                            #
#---------------------------ooo0ooo--------------------------#

###*******FGM-3 Sensor Values**********
mSens = 0.118
cSens = 14.353
###************************************
###************************************

###*******TM36 temp probe conversion  V to degrees C*********
tempScaler=0.0100
tempOffset = 0.5000
###************************************

#-------------create folders if absent------------------------
os.chdir('/home/pi/EFM')
try:
    os.makedirs('Plots/')
except OSError:
    if not os.path.isdir('Plots/'):
        raise
###---------------------------ooo0ooo---------------------------


DataFileLengthSeconds = 86400

# time between samples in seconds      
SamplingPeriod = 2.00


# time between datasaves and plots in seconds, 
DataWriteInterval = 1000

# -- create numpy array to store data
TargetNoSamples = int((DataFileLengthSeconds*1.1)/SamplingPeriod)
DataArray=np.zeros([TargetNoSamples], dtype=np.float32)
nDataPoints = 0
FirstDataReading=True

time.sleep(30)  #give Arduino time to boot
ser = serial.Serial('/dev/ttyACM0', 57600)
ser.flushInput()
time.sleep(2)

StartDateTime=UTCDateTime()
lastSaveTime=StartDateTime

while 1:
  
  time.sleep(SamplingPeriod)
  try:

    freq = np.float32(ser.readline())   # converts the freq number string to a floating point number
    if (freq > 10000):
      #Bfield = -1.0*(((1000000.0/freq)-cSens)/mSens)*1000 # convert freq to field in nTeslas
      Bfield = (((1000000.0/freq)-cSens)/mSens)*1000.0

    else:
      Bfield=0.0
    if (FirstDataReading == False):  #reject first reading which is often spurious
      DataArray[nDataPoints] = Bfield
      nDataPoints = nDataPoints +1
    else:
      FirstDataReading=False # first reading rejected, subsequent will be accepted
      StartDateTime=UTCDateTime()
      lastSaveTime = UTCDateTime()

  except ValueError:
    print ('failed to read serial line correctly --trying again')

      
  if (StartDateTime.day != UTCDateTime().day):
    EndDateTime=UTCDateTime()
    
    threadSaveAndPlot = Thread(target=SaveAndPlot, args=(DataArray, StartDateTime, EndDateTime, nDataPoints,))
    threadSaveAndPlot.start()
    
    DataArray=np.zeros(TargetNoSamples, np.float32)  # zero data array for new day
    StartDateTime=UTCDateTime()
    lastSaveTime=StartDateTime
    nDataPoints = 0
    gc.collect

  if ((UTCDateTime() - lastSaveTime) > DataWriteInterval):
    EndDateTime=UTCDateTime()
    threadSaveAndPlot = Thread(target=SaveAndPlot, args=(DataArray, StartDateTime, EndDateTime, nDataPoints,))
    threadSaveAndPlot.start()
    lastSaveTime = UTCDateTime()


      

