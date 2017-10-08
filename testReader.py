#---------------------------ooo0ooo---------------------------
# import required modules
import numpy as np
import time
import array
import multiprocessing

from array import array
from datetime import datetime
from datetime import timedelta

from scipy import signal

from obspy.imaging.spectrogram import spectrogram
from obspy.signal.filter import bandpass, lowpass, highpass
from obspy.imaging.cm import obspy_sequential
from obspy.signal.tf_misfit import cwt
from obspy import UTCDateTime, read, Trace, Stream
from obspy.signal.trigger import plot_trigger, z_detect

import time
import ftplib
import string

import matplotlib

#matplotlib.use('Agg') #prevent use of Xwindows
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

import matplotlib.dates as mdates
import statistics   #used by median filter
import os
import gc





#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
def bytes_to_int(bytes):
    return int(bytes.encode('hex'), 16)




    #---------------------------ooo0ooo---------------------------
def plotFiltered(Data, samplingFreq, startDT):

    N = len(Data)
    mean_removed = np.ones_like(Data)*np.mean(Data)
    #Data0 = Data - mean_removed
    Data0=Data

    lowCut1 = 0.001
    highCut1 = 2.0
    Data1 =butter_bandpass_filter(Data0, lowCut1, highCut1, samplingFreq, order=3)
    #legend1=

    lowCut2 = 2.0
    highCut2 = 5.0
    Data2 =butter_bandpass_filter(Data0, lowCut2, highCut2, samplingFreq, order=3)

    lowCut3 = 05.0
    highCut3 =10.0
    Data3 =butter_bandpass_filter(Data0, lowCut3, highCut3, samplingFreq, order=3)


    xN=np.linspace(1, (N/samplingFreq), N)
    x = np.divide(xN, 60)
    TitleString=('A: Raw  B:' + str(lowCut1)+'-' +str(highCut1) + ' Hz  C:' + str(lowCut2)+'-' +str(highCut2) + ' Hz  D:' + str(lowCut3)+'-' +str(highCut3) + ' Hz')

    fig=plt.figure()
    fig.canvas.set_window_title('start U.T.C. - '+ startDT)

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(4, 1 )

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, Data0, label='unfiltered')


    ax1 = plt.subplot(gs[1], sharex=ax0) 
    ax1.plot(x, Data1, label= (str(lowCut1) +'-' + str(highCut1) +'Hz'))


    ax2 = plt.subplot(gs[2], sharex=ax0) 
    ax2.plot(x, Data2, label='unfiltered')


    ax3 = plt.subplot(gs[3], sharex=ax0) 
    ax3.plot(x, Data3, label='unfiltered')

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels()
    plt.setp(xticklabels, visible=False)



    fig.tight_layout()
    fig.show()

    #plt.show()
    #---------------------------ooo0ooo---------------------------
def plotBands(tr):
    print('Filtering and plotting bands')

    legendLoc='upper left'
    xMin=0.0
    xMax=24.0

    N = len(tr.data)

    samplingFreq=tr.stats.sampling_rate

    yscale=5.0

    lowCut1 = 0.01
    highCut1 = 1.0
    tr1=tr.copy()
    tr1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)
   
    lowCut2 = 1.0
    highCut2 = 2.0
    tr2=tr.copy()
    tr2.filter('bandpass', freqmin=lowCut2, freqmax=highCut2, corners=4, zerophase=True)

    lowCut3 = 2.0
    highCut3 = 3.0
    tr3=tr.copy()
    tr3.filter('bandpass', freqmin=lowCut3, freqmax=highCut3, corners=4, zerophase=True)

    lowCut4 = 3.0
    highCut4 = 4.0
    tr4=tr.copy()
    tr4.filter('bandpass', freqmin=lowCut4, freqmax=highCut4, corners=4, zerophase=True)

    lowCut5 = 4.0
    highCut5 = 5.0
    tr5=tr.copy()
    tr5.filter('bandpass', freqmin=lowCut5, freqmax=highCut5, corners=5, zerophase=True)

    lowCut6 = 5.0
    highCut6 = 10.0
    tr6=tr.copy()
    tr6.filter('bandpass', freqmin=lowCut6, freqmax=highCut6, corners=4, zerophase=True)

    lowCut7 = 10.0
    highCut7 = 15.0
    tr7=tr.copy()
    tr7.filter('bandpass', freqmin=lowCut7, freqmax=highCut7, corners=4, zerophase=True)


    x=np.linspace(1, (N/tr.stats.sampling_rate), N)
    x = np.divide(x, 3600)

    fig=plt.figure()
    fig.suptitle(str(tr.stats.starttime) + ' Filtered ' )
    fig.canvas.set_window_title('start U.T.C. - '+ str(tr.stats.starttime))


    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(8, 1 )

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr)
    ax0.set_xlim(xMin,xMax)
    ax0.legend(['raw data'], loc=legendLoc, fontsize=10)

    ax1 = plt.subplot(gs[1], sharex=ax0) 
    ax1.plot(x, tr1)
    ax1.legend([str(lowCut1) +'-' + str(highCut1) +'Hz'], loc=legendLoc, fontsize=10)

    ax2 = plt.subplot(gs[2], sharex=ax1) 
    ax2.plot(x, tr2)
    ax2.legend([str(lowCut2) +'-' + str(highCut2) +'Hz'], loc=legendLoc, fontsize=10)       

    ax3 = plt.subplot(gs[3], sharex=ax1) 
    ax3.plot(x, tr3)
    ax3.legend([str(lowCut3) +'-' + str(highCut3) +'Hz'], loc=legendLoc, fontsize=10)
               
    ax4 = plt.subplot(gs[4], sharex=ax1) 
    ax4.plot(x, tr4)
    ax4.legend([str(lowCut4) +'-' + str(highCut4) +'Hz'], loc=legendLoc, fontsize=10)

    ax5 = plt.subplot(gs[5], sharex=ax1) 
    ax5.plot(x, tr5)
    ax5.legend([str(lowCut5) +'-' + str(highCut5) +'Hz'], loc=legendLoc, fontsize=10)
               
    ax6 = plt.subplot(gs[6], sharex=ax1) 
    ax6.plot(x, tr6)
    ax6.legend([str(lowCut6) +'-' + str(highCut6) +'Hz'], loc=legendLoc, fontsize=10)
    
    ax7 = plt.subplot(gs[7], sharex=ax1) 
    ax7.plot(x, tr7)
    ax7.legend([str(lowCut7) +'-' + str(highCut7) +'Hz'], loc=legendLoc, fontsize=10)

   
    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels()+ ax3.get_xticklabels() \
        + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels() 

    plt.setp(xticklabels, visible=False)

    ax7.set_xlabel(r'$\Delta$t - hr', fontsize=12)

    fig.tight_layout()
    fig.show()

    #---------------------------ooo0ooo---------------------------
def plotMultiple(tr1, tr2, tr3, tr4, yMin, yMax, lowCut, highCut):

    N1 = len(tr1.data)
    N2 = len(tr2.data)
    N3 = len(tr3.data)
    N4 = len(tr4.data)

    samplingFreq=(tr1.stats.sampling_rate + tr2.stats.sampling_rate+ tr3.stats.sampling_rate+ tr4.stats.sampling_rate)/4.0

    x1=np.linspace(1, (N1/samplingFreq), N1)
    x1 = np.divide(x1, 3600)

    x2=np.linspace(1, (N2/samplingFreq), N2)
    x2 = np.divide(x2, 3600)
    
    x3=np.linspace(1, (N3/samplingFreq), N3)
    x3 = np.divide(x3, 3600)
    
    x4=np.linspace(1, (N4/samplingFreq), N4)
    x4 = np.divide(x4, 3600)    

    TitleString= str(lowCut) + '-' + str(highCut) + ' Hz'
    fig=plt.figure(figsize=(14,9))
    fig.canvas.set_window_title(TitleString)
    fig.suptitle('Filtered ' + str(lowCut) + '--' + str(highCut) +'Hz')

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(4, 1 )

    ax1 = plt.subplot(gs[0])
    ax1.plot(x1, tr1, color='k')
    ax1.set_xlim([0,24])
    ax1.set_ylim([yMin,yMax])
    ax1.legend([str(tr1.stats.starttime.date)],loc='upper right', fontsize=12)

    ax2 = plt.subplot(gs[1])
    ax2.plot(x2, tr2, color='k')
    ax2.set_xlim([0,24])
    ax2.set_ylim([yMin,yMax])
    ax2.legend([str(tr2.stats.starttime.date)],loc='upper right', fontsize=12)
    ax2.axvline(x=7.4, color='r')
    
    ax3 = plt.subplot(gs[2])
    ax3.plot(x3, tr3, color='k')
    ax3.set_xlim([0,24])
    ax3.set_ylim([yMin,yMax])
    ax3.legend([str(tr3.stats.starttime.date)],loc='upper right', fontsize=12)    
    ax3.axvline(x=17, color='b')

    ax4 = plt.subplot(gs[3], sharex=ax1) 
    ax4.plot(x4, tr4, color='k')
    ax4.set_xlim([0,24])
    ax4.set_ylim([yMin,yMax])
    ax4.legend([str(tr4.stats.starttime.date)],loc='upper right', fontsize=12)
    ax4.set_xlabel(r'$\Delta$t - hr', fontsize=12)

    fig.tight_layout()
    fig.show()


#---------------------------ooo0ooo---------------------------
def butter_bandpass(lowcut, highCut, samplingFreq, order=5):
    nyq = 0.5 * samplingFreq
    low = lowcut / nyq
    high = highCut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
#---------------------------ooo0ooo---------------------------

def butter_bandpass_filter(data, lowcut, highCut, samplingFreq, order=5):
    b, a = butter_bandpass(lowcut, highCut, samplingFreq, order=order)
    y = lfilter(b, a, data)
    return y

#---------------------------ooo0ooo---------------------------

def plotPeriodogram(tr1, fMin, fMax, startT, endT):

    print('plotting Welch Periodogram....')
    Data=tr1.data
    samplingFreq=tr1.stats.sampling_rate
    
    N = len(Data) # Number of samplepoints
    xN=np.linspace(1, (N/samplingFreq), N)
    #xN = np.divide(xN, 60)
    t1 = np.divide(xN, (samplingFreq*60))


    x0=int(startT*samplingFreq)
    x1=int(endT*samplingFreq)
    if (x0 < 0):
        x0=0
    if (x1>N):
        x1=N-1
    subSetLen=x1-x0


    WINDOW_LEN = int((subSetLen/samplingFreq)*1)
    OVERLAP_LEN = WINDOW_LEN / 8

    topX=np.linspace((x0/samplingFreq)+1, (x1/samplingFreq), subSetLen)


    f, Pxx = signal.welch(Data[x0:x1], samplingFreq, scaling='spectrum')
    Pxx=np.divide(Pxx, 60)


    fig=plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1] )

    ax0 = plt.subplot(gs[0])
    ax0.plot(f, Pxx)
    ax0.set_xlim([0.1,fMax])
    ax0.set_xlabel(r'f - Hz', fontsize=14)

    ax1 = plt.subplot(gs[1])
    #ax1.plot(f, Pxx)
    ax1.semilogy(f, Pxx)
    #ax1.set_xlim([0.01,fMax])
    ax1.set_xlabel(r'f - Hz', fontsize=14)
    ax1.set_ylabel(r't - s', fontsize=14)

    xticklabels = ax0.get_xticklabels()
    plt.setp(xticklabels, visible=False)

    fig.tight_layout()
    fig.show()
#---------------------------ooo0ooo---------------------------
def simplePlot(tr, lowcut, highCut):
    tr.plot()

#---------------------------ooo0ooo---------------------------
def plotSpectrogram(tr1,highCut):
    print('plotting Spectrogram....')
    Data=tr1.data
    samplingFreq=tr1.stats.sampling_rate
    N = len(Data) # Number of samplepoints
    xN=np.linspace(1, (N/samplingFreq), N)
    #xN = np.divide(xN, 60)
    t1 = np.divide(xN, (samplingFreq*60))


    fig = plt.figure()
    fig.canvas.set_window_title('FFT Spectrum ' + str(tr1.stats.starttime.date))

    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
    #ax2.set_ylim([1.0,3.0])

#  t = np.arange(spl1[0].stats.N) / spl1[0].stats.sampling_rate
    ax1.plot(xN, Data, 'k')

    #ax,spec = spectrogram(Data, samplingFreq, show=False, axes=ax2)

    ax = spectrogram(Data, samplingFreq, show=False, axes=ax2)
    mappable = ax2.images[0]
    plt.colorbar(mappable=mappable, cax=ax3)

    ax1.set_ylabel(r'$\Delta$ P - Pa')
    ax2.set_ylabel(r'f - Hz', fontsize=14)
    ax2.set_xlabel(r'$\Delta$t - s', fontsize=12)
    #ax2.set_ylim([0.0,6])
    ax2.set_ybound(lower=None, upper=(highCut*2.0))

    fig.show()

#---------------------------ooo0ooo---------------------------
def plotWaveletTransform(tr, startMin, endMin):

    print('Calculating Wavelet Transform')
    N = len(tr.data) # Number of samplepoints
    dt = tr.stats.delta


    x0=int(startMin*tr.stats.sampling_rate*60)
    x1=int(endMin*tr.stats.sampling_rate*60)
    if (x0 < 0):
        x0=0
    if (x1>N):
        x1=N-1
    subSetLen=x1-x0


    t = np.linspace(x0, x1, num=subSetLen)
    t1 = np.divide(t, (tr.stats.sampling_rate*60))
    


    fig = plt.figure()
    fig.suptitle('Wavelet Transform ' + str(tr.stats.starttime.date), fontsize=12)
    fig.canvas.set_window_title('Wavelet Transform ' + str(tr.stats.starttime.date))
    #ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    

    ax1.plot(t1, tr.data[x0:x1], 'k')
    ax1.set_ylabel(r'$\Delta$P - Pa')


    f_min = 0.01
    f_max = 15

    scalogram = cwt(tr.data[x0:x1], dt, 8, f_min, f_max)


    x, y = np.meshgrid(t1, np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))

    ax2.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    
    ax2.set_xlabel("Time  [min]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale('log')
    ax2.set_ylim(f_min, f_max)

    fig.show()
#---------------------------ooo0ooo---------------------------
def plot1hrWaveletTransform(tr1, startHour):
    startMin=startHour*60
    endMin=startMin+60

    plotWaveletTransform(tr1, startMin, endMin)

#---------------------------ooo0ooo---------------------------

def plotDayplot(tr,lowCut, highCut):
    
    tr.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    tr.plot(type="dayplot", interval=60, right_vertical_labels=False, one_tick_per_line=True, color=['k', 'r', 'b', 'g'], show_y_UTC_label=True)

#---------------------------ooo0ooo---------------------------
def CalcRunningMeanPower(tr, deltaT, lowCut, highCut):

    N=len(tr)
    dt = tr.stats.delta
    newStream=tr.copy()
    

    newStream.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    x=newStream.data
   
    x=x**2
    
    nSamplePoints=int(deltaT/dt)
    runningMean=np.zeros((N-nSamplePoints), np.float32)

    #determie first tranche
    tempSum = 0.0
    
    for i in range(0,(nSamplePoints-1)):
        tempSum = tempSum + x[i]

        runningMean[i]=tempSum

    #calc rest of the sums by subracting first value and adding new one from far end  
    for i in range(1,(N-(nSamplePoints+1))):
            tempSum = tempSum - x[i-1] + x[i + nSamplePoints]
            runningMean[i]=tempSum
    # calc averaged acoustic intensity as P^2/(density*c)
    runningMean=runningMean/(1.2*330)


    newStream.data=runningMean
    newStream.stats.npts=len(runningMean)

    return newStream
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
def z_DayPlotAcousticPower(tr, deltaT, lowCut, highCut):
    
    st2 = CalcRunningMeanPower(tr, deltaT, lowCut, highCut)
    st2.plot(type="dayplot", interval=60, right_vertical_labels=False, one_tick_per_line=True, color=['k', 'r', 'b', 'g'], show_y_UTC_label=True)
    
#---------------------------ooo0ooo---------------------------
def z_plotMany():
    st1 = read("Data/Day_17.mseed")
    st2 = read("Data/Day_18.mseed")
    st3 = read("Data/Day_19.mseed")
    st4 = read("Data/Day_20.mseed")
    #st.plot()
    st1.detrend(type='demean')
    st2.detrend(type='demean')
    st3.detrend(type='demean')
    st4.detrend(type='demean')

    tr1=st1[0].copy()
    tr2=st2[0].copy()
    tr3=st3[0].copy()
    tr4=st4[0].copy()

    lowCut = 0.05
    highCut = 2.0
    yMax=30
    yMin=0

    #tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    #tr2.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    #tr3.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    #tr4.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    
    deltaT=5.0
    tr11=CalcRunningMeanPower(tr1, deltaT, lowCut, highCut)
    tr12=CalcRunningMeanPower(tr2, deltaT, lowCut, highCut)
    tr13=CalcRunningMeanPower(tr3, deltaT, lowCut, highCut)
    tr14=CalcRunningMeanPower(tr4, deltaT, lowCut, highCut)

    plotMultiple(tr11, tr12, tr13, tr14, yMin, yMax, lowCut, highCut)

#---------------------------ooo0ooo---------------------------
def z_plotPeriodgram(tr,lowCut, highCut):
    

    yscale=highCut*2.0
    tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    plotPeriodogram(tr, 0.01, 15.0, 0, 1000000)

#---------------------------ooo0ooo---------------------------
def z_wavelets(tr1,lowCut, highCut):


    tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    plotWaveletTransform(tr1, 420, 480)

#---------------------------ooo0ooo---------------------------
def z_spectrum(tr1,lowCut, highCut):

    tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
    plotSpectrogram(tr1,highCut)

#---------------------------ooo0ooo---------------------------
def z_plotBands(tr):
                                                                                                                                                                                                                                                                                                                                                                                                                    
    plotBands(tr)
#---------------------------ooo0ooo---------------------------
def z_plotDayPlot(tr,lowCut, highCut):
                                                                                                                                                                                                                                                                                                                                                                                                                       
    plotDayplot(tr,lowCut, highCut)
#---------------------------ooo0ooo---------------------------
def z_simplePlot(tr,lowCut, highCut):
    tr.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
                                                                                                                                                                                                                                                                                                                                                                                                                            
    simplePlot(tr, lowCut, highCut)
#---------------------------ooo0ooo---------------------------

st1 = read("Data/Day_2.mseed")
st1.detrend(type='demean')

tr2=st1[0].copy()

lowCut=0.01
highCut=10.0


#z_plotMany()
#z_plotPeriodgram((tr,lowCut, highCut))

#z_plotBands(tr2)

tr=st1[0].copy()
z_plotDayPlot(tr,lowCut, highCut)

tr=st1[0].copy()
z_simplePlot(tr,lowCut, highCut)


z_DayPlotAcousticPower(tr, 5.0, lowCut, highCut)
#z_wavelets(tr1,lowCut, highCut)

tr=st1[0].copy()
z_spectrum(tr,lowCut, highCut)

        

#plotDayplot(st,0.01, 15.0)
#plotDayplot(st2,0.01, 15.0)
#plotDayplot(st,0.01, 5.0)


#
#df = tr1.stats.sampling_rate
#cft = z_detect(tr1.data, int(10 * df))
#
#plot_trigger(tr1, cft, -0.4, -0.3)


#plotBands(st)

#fig=plt.figure()
#fig.canvas.set_window_title('start U.T.C. - '+ str(st[0].stats.starttime))
#
#plt.subplots_adjust(hspace=0.001)
#gs = gridspec.GridSpec(2, 1 )
#
#ax0 = plt.subplot(gs[0])
#ax0.plot(tr1, label='unfiltered')
#ax0.legend('Raw Data', loc='upper right')
#
#
#ax1 = plt.subplot(gs[1], sharex=ax0) 
#ax1.plot(tr2, label='unfiltered')
#
#
#
#xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels()
#plt.setp(xticklabels, visible=False)
#
#ax1.set_xlabel(r'$\Delta$t - min', fontsize=12)
#
#
#
#fig.tight_layout()
#fig.show()





#tr1.spectrogram(title='IR.IfS-1  ' + str(st[0].stats.starttime))




#plotSpectrogram(tr1, 15.0)

#startHour=1.0
#startHour=float(input('input start hour for wavelet transform....'))

#plot1hrWaveletTransform(tr1, startHour)
#period=float(input('input period (min) for wavelet transform....'))
#startMin=startHour*60.0
#endMin=startMin + period
#plotWaveletTransform(tr1, startMin, endMin)
#input()

#plotSpectrogram(Data, samplingFreq, lowCut, highCut, startT, endT,startDT)
#plotPeriodogram(DataFiltered, samplingFreq, lowCut, highCut, startT, endT)



#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------
#---------------------------ooo0ooo---------------------------

#def plotBands(st):
#
#    N = len(st[0].data)
#
#    samplingFreq=st[0].stats.sampling_rate
#
#    Data0=st[0].data
#
#    lowCut1 = 0.001
#    highCut1 = 2.0
#    Data1= st[0]
#    #Data1 = bandpass(st[0].data, lowCut1, highCut1, samplingFreq, corners=4, zerophase=True)
#    #Data1.filter('bandpass', lowCut1, highCut1, samplingFreq, corners=4, zerophase=True)
#    Data1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)
#
#
#    lowCut1 = 0.001
#    highCut1 = 2.0
#    Data1 = bandpass(st[0].data, lowCut1, highCut1, samplingFreq, corners=4, zerophase=True)
#   
#    lowCut2 = 2.0
#    highCut2 = 5.0
#    Data2= bandpass(st[0].data, lowCut2, highCut2, samplingFreq, corners=4, zerophase=True)
#
#    lowCut3 = 05.0
#    highCut3 = 10.0
#    Data3= bandpass(st[0].data, lowCut3, highCut3, samplingFreq, corners=4, zerophase=True)
#
#    lowCut4 = 10.0
#    highCut4 = 18.0
#    Data4= bandpass(st[0].data, lowCut4, highCut4, samplingFreq, corners=4, zerophase=True)
#
#
#    x=np.linspace(1, (N/st[0].stats.sampling_rate), N)
#    x = np.divide(x, 60)
#
#    TitleString=('A: Raw  B:' + str(lowCut1)+'-' +str(highCut1) + ' Hz  C:' + str(lowCut2)+'-'
#               +str(highCut2) + ' Hz  D:' + str(lowCut3)+'-' +str(highCut3) + ' Hz  E'+ str(lowCut4)+'-' +str(highCut4) + ' Hz')
#
#    fig=plt.figure()
#    #fig.canvas.set_window_title('start U.T.C. - '+ startDT)
#    fig.canvas.set_window_title('start U.T.C. - '+ str(st[0].stats.starttime))
#
#
#    plt.subplots_adjust(hspace=0.001)
#    gs = gridspec.GridSpec(5, 1 )
#
#    ax0 = plt.subplot(gs[0])
#    ax0.plot(x, Data0, label='unfiltered')
#    #ax0.legend('Raw Data', loc='upper right')
#
#
#    ax1 = plt.subplot(gs[1], sharex=ax0) 
#    ax1.plot(x, Data1, label= (str(lowCut1) +'-' + str(highCut1) +'Hz'))
#
#
#    ax2 = plt.subplot(gs[2], sharex=ax0) 
#    ax2.plot(x, Data2, label='unfiltered')
#
#
#    ax3 = plt.subplot(gs[3], sharex=ax0) 
#    ax3.plot(x, Data3, label='unfiltered')
#
#    ax4 = plt.subplot(gs[4], sharex=ax0) 
#    ax4.plot(x, Data4, label='unfiltered')
#
#    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels()
#    plt.setp(xticklabels, visible=False)
#
#    ax4.set_xlabel(r'$\Delta$t - min', fontsize=12)
#
#
#
#    fig.tight_layout()
#    fig.show()
#
#    #---------------------------ooo0ooo---------------------------

    #---------------------------ooo0ooo---------------------------
#def plotRaw(Data, samplingFreq):
#  plt.plot(Data)
#  plt.ylabel('Pressure Pa')
#  plt.grid(True)
#
#    mean_removed = np.ones_like(Data)*np.mean(Data)
#    Data2 = Data - mean_removed
#
#    lowCut = 0.2
#    highCut = 10.0 
#    Data2 =butter_bandpass_filter(Data, lowCut, highCut, samplingFreq, order=3)
#
#    fig=plt.figure(figsize=(8, 8))
#
#    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1] )
#
#    ax0 = plt.subplot(gs[0])
#    ax0.plot(Data)
#    ax0.set_ylabel(r'Pressure Pa')
#
#    ax1 = plt.subplot(gs[1]) 
#    ax1.plot(Data2)
#    ax1.set_ylabel(r'Pressure Pa')
#
#    fig.tight_layout()
#    fig.show()
#
#    plt.show()



#def plotSpectrogram(tr1):
#    
#    Data=tr1[0].data
#    
#
#    N = len(Data) # Number of samplepoints
#    xN=np.linspace(1, (N/st[0].stats.sampling_rate), N)
#    #xN = np.divide(xN, 60)
#    t1 = np.divide(xN, (st[0].stats.sampling_rate*60))
#    #tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
#
#
#    fig = plt.figure()
#    fig.canvas.set_window_title('start U.T.C. - '+ str(st[0].stats.starttime))
#    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
#    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
#    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])
#    #ax2.set_ylim([1.0,3.0])
#
#    
#  t = np.arange(spl1[0].stats.N) / spl1[0].stats.sampling_rate
#    ax1.plot(xN, Data, 'k')
#
#    #ax,spec = spectrogram(Data, samplingFreq, show=False, axes=ax2)
#
#    ax = spectrogram( Data, st[0].stats.sampling_rate, log=True, show=False, axes=ax2)
#    mappable = ax2.images[0]
#    plt.colorbar(mappable=mappable, cax=ax3)
#
#    ax1.set_ylabel(r'$\Delta$ P - Pa')
#    ax2.set_ylabel(r'f - Hz', fontsize=14)
#    ax2.set_xlabel(r'$\Delta$t - s', fontsize=12)
#    #ax2.set_ylim([0.0,10])
#    ax2.set_ybound(lower=None, upper=10.0)
#
#    fig.show()
#---------------------------ooo0ooo---------------------------
#def conVertToMiniSeed(Data):
#    # Convert to NumPy character array
#    # data1 = np.fromstring(Data, dtype='|S1')
#    data1=Data
#    # Fill header attributes
#    stats = {'network': 'BW', 'station': 'RJOB', 'location': '',
#           'channel': 'WLZ', 'npts': len(data1), 'sampling_rate': 0.1,
#           'mseed': {'dataquality': 'D'}}
#    # set current time
#    stats['starttime'] = UTCDateTime()
#    st = Stream([Trace(data=data1, header=stats)])
#    # write as ASCII file (encoding=0)
#    st.write("weather.mseed",format='MSEED',  encoding=4, reclen=256)
#
#    # Show that it worked, convert NumPy character array back to string
#    st1 = read("weather.mseed")
#
#    print(st1[0].stats)
#    new1=np.array(st[0].data)
#
#    plotBands(new1, 38.0, 0)

