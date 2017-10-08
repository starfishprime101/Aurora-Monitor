from obspy import UTCDateTime, read, Trace, Stream

import numpy as np


import matplotlib
#matplotlib.use('Agg') #prevent use of Xwindows
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import matplotlib.dates as mdates
import statistics   #used by median filter
import os
import gc

#---------------------------ooo0ooo---------------------------


def plotSimpleGraph(DataArray):
    data=DataArray
    nDataPoints=len(data)
    
    if nDataPoints>10:
     
        zeroPoint = data[1]
        data = data-zeroPoint

        YAxisRange=np.amax(data)*1.1

        xvalues=np.linspace(0,(nDataPoints-1),nDataPoints)

        
        #fig = plt.figure()


        zeroLabel=str('%0.5g' % (zeroPoint/1000))
        plt.ylabel('$\Delta$Flux Density - nT     0.0='+zeroLabel+'$\mu T$')
        #axes.relim()
        #plt.ylim(((-YAxisRange),(+YAxisRange)))
        plt.plot(xvalues, data,  marker='None',    color = 'darkolivegreen')
      

        plt.grid(True)

        #plt.savefig('simplePlot.svg')
        plt.show()
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
def displayData(ydata):

  z=len(ydata)
  for j in range(0, (z-10), 10):
    for i in range(0,10):
        print ((i+j), '....', ydata[i+j])
    input()
#---------------------------ooo0ooo---------------------------
#st = read("Data/2017/2017_8_8.mseed")
st = read("Data/2017/2017_8_27.mseed")
#st[0].plot()
ydata=st[0].data

plotSimpleGraph(ydata)
input()


ydata=removeGlitches(ydata)
print('plotting deglitched')
plotSimpleGraph(ydata)
input()





print('ploting median filtered')
ydata=medianFilter(ydata)
plotSimpleGraph(ydata)
input()



displayData(ydata)
##st[0].data=ydata
##st[0].plot()
##print(st[0].stats)
##








##DataArray=st[0].data
##StartDateTime=st[0].stats.starttime
##
##nDataPoints=st[0].stats.npts
##deltaT = st[0].stats.delta
##print(StartDateTime.date)
##
##EndDateTime = UTCDateTime()
##
##
##plotTodaysGraph(DataArray, StartDateTime, nDataPoints, deltaT)
###st.plot(size=(800, 600), outfile='singlechannel.png')
##st[0].plot()
###print(st[0].data)
##
##tr=st[0].copy()
                 
