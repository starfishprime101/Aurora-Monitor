
# ---------------------------ooo0ooo---------------------------
# import required modules
import numpy as np
import time
import array
import multiprocessing
import obspy

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
from obspy.io.xseed import Parser
from obspy.signal import PPSD
from obspy.signal.cross_correlation import xcorr_pick_correction

import time
import string

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#import matplotlib.dates as mdates
import statistics  # used by median filter
import os
import gc
from tkinter import filedialog
from tkinter import *
from tkinter import filedialog
# from tkinter import * #for file opening dialog
import tkinter as tk
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# ------------------------------------------------------#
#                                  						#
#             		Function declarations            	#
#                                  						#
# ------------------------------------------------------#

# ---------------------------ooo0ooo---------------------------
def readDataFile():
    root = tk.Tk()
    root.withdraw()
    root.filename = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                               filetypes=[("miniseed data files", "*.mseed")])
    st = read(root.filename)
    return st

# ---------------------------ooo0ooo---------------------------
def readDataFolder(resamplefreq):
    st = Stream()
    root = Tk()
    root.withdraw()
    folderSelected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select directory")

    # Read in all files within slected directory.
    listing = os.listdir(folderSelected)
    for file in listing:
        if '.mseed' in file:
            print(file)
            streamTemp=obspy.read(folderSelected+'/'+file)
            streamTemp.resample(resamplefreq) 
            st += streamTemp
    st.sort(['starttime'])
    st.merge(method=1, fill_value="interpolate")
    #st.merge(method=1, fill_value=0.00)

    return st

# ---------------------------ooo0ooo---------------------------
def simplePlot(tr):
    fig=tr.plot()
    fig.savefig("simplePlot.png")
    return

# ---------------------------ooo0ooo---------------------------
def plotDayplot(tr, lowCut, highCut):
	

	plotTitle='Raw Pressure :: '+str(tr.stats.station)+'-'+str(tr.stats.channel)+'-'+str(tr.stats.location)+' :: '+str(tr.stats.starttime.date)\
	+' :: '+str(lowCut)+'-'+str(highCut)+'Hz'
	fig=tr.plot(type="dayplot", title=plotTitle, data_unit='$\Delta$Pa', interval=60, right_vertical_labels=False,\
	one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
	fig.savefig("rawDayPlot.png")
	return

# ---------------------------ooo0ooo---------------------------
def plotBands(tr,deltaT):

    N = len(tr.data)

    samplingFreq = tr.stats.sampling_rate
    timeInsecs= len(tr.data)/samplingFreq


    tr0b = CalcRunningMeanPower(tr, deltaT)
    
    lowCut1 = 0.01
    highCut1 = 20.0
    tr1 = tr.copy()
    tr1.filter('bandpass', freqmin=lowCut1, freqmax=highCut1, corners=4, zerophase=True)
    tr1b = CalcRunningMeanPower(tr1, deltaT)
    
    
    lowCut2 = 0.01
    highCut2 = 0.5
    tr2 = tr.copy() ##must use a copy others filers are overlaid on same data
    tr2.filter('bandpass', freqmin=lowCut2, freqmax=highCut2, corners=4, zerophase=True)
    tr2b = CalcRunningMeanPower(tr2, deltaT)

    lowCut3 = 0.5
    highCut3 = 2.5
    tr3 = tr.copy()
    tr3.filter('bandpass', freqmin=lowCut3, freqmax=highCut3, corners=4, zerophase=True)
    tr3b = CalcRunningMeanPower(tr3, deltaT)
    
    lowCut4 = 2.5
    highCut4 = 5.0
    tr4 = tr.copy()
    tr4.filter('bandpass', freqmin=lowCut4, freqmax=highCut4, corners=4, zerophase=True)
    tr4b = CalcRunningMeanPower(tr4, deltaT)
    
    lowCut5 = 5.0
    highCut5 = 10.0
    tr5 = tr.copy()
    tr5.filter('bandpass', freqmin=lowCut5, freqmax=highCut5, corners=4, zerophase=True)
    tr5b = CalcRunningMeanPower(tr5, deltaT)

    lowCut6 = 10.0
    highCut6 = 20.0
    tr6 = tr.copy()
    tr6.filter('bandpass', freqmin=lowCut6, freqmax=highCut6, corners=5, zerophase=True)
    tr6b = CalcRunningMeanPower(tr6, deltaT)
 
    
    print('Filtering and plotting raw signal bands')
    legendLoc = 'best'
   
    x = np.linspace(1, (N / tr.stats.sampling_rate), N)
    x = np.divide(x, 3600)
    
    fig = plt.figure()
#    fig = plt.figure(figsize=(8, 6), dpi=300)
    fig.suptitle(str(tr.stats.starttime) + ' Filtered ')
    fig.canvas.set_window_title('start U.T.C. - ' + str(tr.stats.starttime))

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(7, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr)
    ax0.legend(['raw data'], loc=legendLoc, fontsize=7)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, tr1)
    ax1.legend([str(lowCut1) + '-' + str(highCut1) + 'Hz'], loc=legendLoc, fontsize=7)
    #ax1.title.set_text(str(lowCut1) + '-' + str(highCut1) + 'Hz')

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(x, tr2)
    ax2.legend([str(lowCut2) + '-' + str(highCut2) + 'Hz'], loc=legendLoc, fontsize=7)
    
    ax3 = plt.subplot(gs[3], sharex=ax1)
    ax3.plot(x, tr3)
    ax3.legend([str(lowCut3) + '-' + str(highCut3) + 'Hz'], loc=legendLoc, fontsize=7)

    ax4 = plt.subplot(gs[4], sharex=ax1)
    ax4.plot(x, tr4)
    ax4.legend([str(lowCut4) + '-' + str(highCut4) + 'Hz'], loc=legendLoc, fontsize=7)

    ax5 = plt.subplot(gs[5], sharex=ax1)
    ax5.plot(x, tr5)
    ax5.legend([str(lowCut5) + '-' + str(highCut5) + 'Hz'], loc=legendLoc, fontsize=7)
    
    ax6 = plt.subplot(gs[6], sharex=ax1)
    ax6.plot(x, tr6)
    ax6.legend([str(lowCut6) + '-' + str(highCut6) + 'Hz'], loc=legendLoc, fontsize=7)

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels() + ax3.get_xticklabels() \
                  + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels()

    plt.setp(xticklabels, visible=False)

    ax6.set_xlabel(r'$\Delta$t - hr', fontsize=10)

    fig.tight_layout()
    fig.savefig("rawSignalBands.png")
    plt.show()
    
    print('Filtering and plotting acoustic pwr bands')
    N = len(tr0b.data)
    x = np.linspace(1, timeInsecs, N)
    x = np.divide(x, 3600)

    fig = plt.figure()
    fig.suptitle(str(tr.stats.starttime) + 'Acoustic Power Bands')
    fig.canvas.set_window_title('start U.T.C. - ' + str(tr.stats.starttime))

    plt.subplots_adjust(hspace=0.001)
    gs = gridspec.GridSpec(7, 1)

    ax0 = plt.subplot(gs[0])
    ax0.plot(x, tr0b)

    ax0.legend(['raw data'], loc=legendLoc, fontsize=7)

    ax1 = plt.subplot(gs[1], sharex=ax0)
    ax1.plot(x, tr1b)
    ax1.legend([str(lowCut1) + '-' + str(highCut1) + 'Hz'], loc=legendLoc, fontsize=7)

    ax2 = plt.subplot(gs[2], sharex=ax1)
    ax2.plot(x, tr2b)
    ax2.legend([str(lowCut2) + '-' + str(highCut2) + 'Hz'], loc=legendLoc, fontsize=7)

    ax3 = plt.subplot(gs[3], sharex=ax1)
    ax3.plot(x, tr3b)
    ax3.legend([str(lowCut3) + '-' + str(highCut3) + 'Hz'], loc=legendLoc, fontsize=7)

    ax4 = plt.subplot(gs[4], sharex=ax1)
    ax4.plot(x, tr4b)
    ax4.legend([str(lowCut4) + '-' + str(highCut4) + 'Hz'], loc=legendLoc, fontsize=7)

    ax5 = plt.subplot(gs[5], sharex=ax1)
    ax5.plot(x, tr5b)
    ax5.legend([str(lowCut5) + '-' + str(highCut5) + 'Hz'], loc=legendLoc, fontsize=7)

    ax6 = plt.subplot(gs[6], sharex=ax1)
    ax6.plot(x, tr6b)
    ax6.legend([str(lowCut6) + '-' + str(highCut6) + 'Hz'], loc=legendLoc, fontsize=7)

    xticklabels = ax0.get_xticklabels() + ax1.get_xticklabels() + ax2.get_xticklabels() + ax3.get_xticklabels() \
                  + ax4.get_xticklabels() + ax5.get_xticklabels() + ax6.get_xticklabels()

    plt.setp(xticklabels, visible=False)

    ax6.set_xlabel(r'$\Delta$t - hr', fontsize=10)

    fig.tight_layout()
    fig.savefig("acousticPwrBands.png")
    plt.show()
    input("Press Enter to continue...")


# ---------------------------ooo0ooo---------------------------
def plotAcousticPower(tr, deltaT):
    st2 = CalcRunningMeanPower(tr, deltaT)
    fig=st2.plot(title='test', data_unit='$Wm^{-2}$', show_y_UTC_label=False)
    fig.savefig("AcousticPwr.png")
    return


# ---------------------------ooo0ooo---------------------------
def plotDayPlotAcousticPower(tr, deltaT, lowCut, highCut):
	

	plotTitle='Acoustic Power :: '+str(tr.stats.station)+'-'+str(tr.stats.channel)+'-'+str(tr.stats.location)+' :: '+str(tr.stats.starttime.date)\
	+' :: '+str(lowCut)+'-'+str(highCut)+'Hz'

	st2 = CalcRunningMeanPower(tr, deltaT)
	fig=st2.plot(type="dayplot", title=plotTitle, data_unit='$Wm^{-2}$', interval=60, right_vertical_labels=False,
	one_tick_per_line=False, color=['k', 'r', 'b', 'g'], show_y_UTC_label=False)
	fig.savefig("DayPlotAcousticPwr.png")
	return

# ---------------------------ooo0ooo---------------------------
def plotWelchPeriodogram(tr, fMin, fMax):
    print('plotting Welch Periodogram....')
    Data = tr.data
    samplingFreq = tr.stats.sampling_rate

    N = len(Data)  # Number of samplepoints
    xN = np.linspace(1, (N / samplingFreq), N)
    # xN = np.divide(xN, 60)
    t1 = np.divide(xN, (samplingFreq * 60))

    x0 = 0
    x1 = N - 2
    if (x0 < 0):
        x0 = 0
    if (x1 > N):
        x1 = N - 1
    subSetLen = x1 - x0

    WINDOW_LEN = int((subSetLen / samplingFreq) * 1)
    OVERLAP_LEN = WINDOW_LEN / 8

    topX = np.linspace((x0 / samplingFreq) + 1, (x1 / samplingFreq), subSetLen)

    f, Pxx = signal.welch(Data[x0:x1], samplingFreq, nperseg=2000, scaling='density')
    Pxx = np.divide(Pxx, 60)

    fig, ax = plt.subplots()

    ax.set_title("Welch Power Density Log Spectrum")
    ax.grid()

    ax.plot(f, Pxx)
    ax.semilogy(f, Pxx)
    ax.set_xscale('log')
    #ax.set_xlim([0.0, fMax])
    #ax.set_xlim([0.0, fMax - 2.0])
    ax.set_xlabel(r'f - Hz', fontsize=14)
    ax.set_ylabel(r'Relative Power Amplitude', fontsize=14)

    fig.tight_layout()
    fig.savefig("WelchPeriodogram.png")

    plt.show()

# ---------------------------ooo0ooo---------------------------
def plotSpectrogram1(tr):
    print('plotting Spectrogram.1...')

    tr.spectrogram(samp_rate=tr.stats.sampling_rate ,log=True, title='FFT Spectrum ' + str(tr.stats.starttime.date)+ str(tr.stats.starttime.time))


# ---------------------------ooo0ooo---------------------------
def plotSpectrogram2(tr, lowCut, highCut):
    print('plotting Spectrogram.2...')

    beg = tr.stats.starttime    
    end = tr.stats.endtime

    #setup figure
    fig = plt.figure()


    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2]) #[left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)
    ax3 = fig.add_axes([0.83, 0.1, 0.03, 0.6])

    #make time vector
    t = np.arange(tr.stats.npts) / tr.stats.sampling_rate

    #plot waveform (top subfigure)    
    ax1.plot(t, tr.data, 'k')

   #plot spectrogram (bottom subfigure)
    tr1 = tr
    fig = tr.spectrogram(show=False, axes=ax2)

    mappable = ax2.images[0]
    plt.colorbar(mappable=mappable, cax=ax3)

    ax2.set_ylim(lowCut, highCut)

    plt.show()
    #print some info
    print('Figure starttime = %s'%beg)
    print('Figure endtime   = %s'%end)

# ---------------------------ooo0ooo---------------------------
def plotSpectrogram3(tr, lowCut, highCut):
    print('plotting Spectrogram.3...')

    samplingFreq = tr.stats.sampling_rate

    # ~ fig = tr.spectrogram(show=False)
    # ~ ax = fig.axes[0]
    # ~ mappable = ax.images[0]
    # ~ plt.colorbar(mappable=mappable, ax=ax)
    
    fig = tr.spectrogram(show=False, log=True)
    ax = fig.axes[0]
    mappable = ax.collections[0]
    plt.colorbar(mappable=mappable, ax=ax)

    plt.show()


# ---------------------------ooo0ooo---------------------------
def plotSpectrogram4(tr):
	print('plotting Spectrogram.4...')
	NFFT = 512  # length of spectrogram window in sample points (initial: 256)
	# number of sample points that the sliding window overlaps, must be less than NFFT
	noverlap = 50  
	xstart = 0    # x axis limits in the plot
	xend = 3600  # max. length of signal: 21627 sec
	
	# plot
	ax1 = plt.subplot(211)
	plt.plot(tr.times(), tr.data, linewidth=0.5)
	plt.xlabel('time [sec]')
	plt.ylabel('pressure [Pa]')
	
	plt.subplot(212, sharex=ax1)
	plt.title('spectrogram, window length %s pts' % NFFT)
	Pxx, freqs, bins, im = plt.specgram(
	    tr.data, NFFT=NFFT, Fs=tr.stats.sampling_rate, 
	    noverlap=noverlap,cmap=plt.cm.gist_heat)
	
	# Pxx is the segments x freqs array of instantaneous power, freqs is
	# the frequency vector, bins are the centers of the time bins in which
	# the power is computed, and im is the matplotlib.image.AxesImage instance
	plt.ylabel('frequency [Hz]')
	plt.xlabel('time [sec]')
	plt.ylim(0,20.0)
	
	plt.xlim(xstart, xend)
	plt.show()
# ---------------------------ooo0ooo---------------------------
def plotWaveletTransform(tr, lowCut, highCut):
    print('Calculating Wavelet Transform')
    plotTitle=str(tr.stats.station)+'-'+str(tr.stats.channel)+'-'+str(tr.stats.location)+' :: '+str(tr.stats.starttime)+':'\
	+' '+str(lowCut)+'-'+str(highCut)+'Hz'
    N = len(tr.data)  # Number of samplepoints
    dt = tr.stats.delta

    x0 = 0
    x1 = N - 1

    t = np.linspace(x0, x1, num=N)
    t1 = np.divide(t, (tr.stats.sampling_rate * 60))

    fig = plt.figure()
    fig.suptitle(plotTitle, fontsize=7)
    fig.canvas.set_window_title(plotTitle)
    # ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60])
    ax1 = fig.add_axes([0.1, 0.75, 0.7, 0.2])  # [left bottom width height]
    ax2 = fig.add_axes([0.1, 0.1, 0.7, 0.60], sharex=ax1)

    print("x1", x1, "len t", len(t), "len t1", len(t1))

    ax1.plot(t1, tr.data, 'k')
    ax1.set_ylabel(r'$\Delta$P - Pa')

    scalogram = cwt(tr.data[x0:x1], dt, 8, lowCut, highCut)

    x, y = np.meshgrid(t1, np.logspace(np.log10(lowCut), np.log10(highCut), scalogram.shape[0]))

    ax2.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)

    ax2.set_xlabel("Time  [min]")
    ax2.set_ylabel("Frequency [Hz]")
    ax2.set_yscale('log')
    ax2.set_ylim(lowCut, highCut)
    #fig.savefig("wavelet.png")
    fig.show()
    input("Press Enter to continue...")

# ---------------------------ooo0ooo---------------------------
def plotWaveletTransformBeta(tr, lowCut, highCut):
    print('Calculating Wavelet Transform Beta')
    npts = tr.stats.npts
    dt = tr.stats.delta
    t = np.linspace(0, dt * npts, npts)
    f_min = lowCut
    f_max = highCut
    
    scalogram = cwt(tr.data, dt, 8, f_min, f_max)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    x, y = np.meshgrid(t,np.logspace(np.log10(f_min), np.log10(f_max), scalogram.shape[0]))
    ax.pcolormesh(x, y, np.abs(scalogram), cmap=obspy_sequential)
    ax.set_xlabel("Time after %s [s]" % tr.stats.starttime)
    ax.set_ylabel("Frequency [Hz]")
    ax.set_yscale('log')
    ax.set_ylim(f_min, f_max)
    plt.show()
    input("Press Enter to continue...")


# ---------------------------ooo0ooo---------------------------
def plotPwrMagnitudeSpectrumLin(tr):
    print('plotting magnitude spectrum....')
    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime
    deltaTime=(tr.stats.endtime-tr.stats.starttime)
    print(deltaTime)

    t = np.arange(0, deltaTime, dt)  # create np array for time axis

    sigTemp = tr.data
    s = sigTemp[0:len(t)]

    fig, ax = plt.subplots()

    # plot spectrum types:
    ax.set_title("Magnitude Spectrum")
    ax.grid()
    ax.magnitude_spectrum(s, Fs=Fs, color='C1')

    fig.savefig("pwrLin.png")

    plt.show()
    return
    
# ---------------------------ooo0ooo---------------------------
def plotPwrMagnitudeSpectrumLog(tr):
    print('plotting magnitude spectrum....')
    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime
    deltaTime=(tr.stats.endtime-tr.stats.starttime)
    print(deltaTime)

    t = np.arange(0, deltaTime, dt)  # create np array for time axis

    sigTemp = tr.data
    s = sigTemp[0:len(t)]

    fig, ax = plt.subplots()

    # plot spectrum types:
    ax.set_title("Magnitude Spectrum")
    ax.grid()
    ax.magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    fig.savefig("pwrLog.png")

    plt.show()
    return
# -----------
# ---------------------------ooo0ooo---------------------------
def plotSimpleFFT(tr):
	# experimental
	fNy = 45.0  #Nyquist Frequency)
	dt=tr.stats.delta
	signal=tr.data
	n=len(signal)
	
	
	
	y_f = np.fft.rfft(signal)
	freq = np.linspace(0, fNy, len(y_f))  
	plt.plot(freq[:len(y_f)], abs(y_f), 'g--', label="Downsample with lowpass", lw=3)
	plt.show()
	
	transformed = np.fft.fft(signal,n)
	shifted = np.fft.fftshift(transformed)
	newSignal=np.fft.ifft(shifted)
	freq = (1/(dt*n)) * np.arange(n)
	plt.plot(signal)
	plt.show()
	plt.plot(newSignal)
	plt.show()
	plt.plot(transformed, lw=2, label="Transformed")
	plt.plot(shifted, '--', lw=3, label="Shifted")
	plt.title('Transformed cosine')
	plt.xlabel('Frequency')
	plt.ylabel('Amplitude')
	plt.grid()
	plt.show()
	return
# ---------------------------ooo0ooo---------------------------	
def plotFFT(tr):	
	## Compute Fourier Transform
	
	delta=tr.stats.delta
	signal=tr.data
	n = len(signal)
	t = tr.times("utcdatetime") 
	minsignal, maxsignal = signal.min(), signal.max()
	
	
	fhat = np.fft.fft(signal, n) #computes the fft
	psd = fhat * np.conj(fhat)/n
	freq = (1/(delta*n)) * np.arange(n) #frequency array
	idxs_half = np.arange(1, np.floor(n/2), dtype=np.int32) #first half index
	psd_real = np.abs(psd[idxs_half]) #amplitude for first half
	
	
	## Filter out noise
	sort_psd = np.sort(psd_real)[::-1]
	# print(len(sort_psd))
	threshold = sort_psd[300]
	psd_idxs = psd > threshold #array of 0 and 1
	psd_clean = psd * psd_idxs #zero out all the unnecessary powers
	fhat_clean = psd_idxs * fhat #used to retrieve the signal
	
	signal_filtered = np.fft.ifft(fhat_clean) #inverse fourier transform
	
	
	## Visualization
	fig, ax = plt.subplots(4,1)
	ax[0].plot(t, signal, color='b', lw=0.5, label='Noisy Signal')
	ax[0].set_xlabel('t axis')
	ax[0].set_ylabel('Accn in Gal')
	ax[0].legend()
	
	ax[1].plot(freq[idxs_half], np.abs(psd[idxs_half]), color='b', lw=0.5, label='PSD noisy')
	ax[1].set_xlabel('Frequencies in Hz')
	ax[1].set_ylabel('Amplitude')
	ax[1].legend()
	
	ax[2].plot(freq[idxs_half], np.abs(psd_clean[idxs_half]), color='r', lw=1, label='PSD clean')
	ax[2].set_xlabel('Frequencies in Hz')
	ax[2].set_ylabel('Amplitude')
	ax[2].legend()
	
	ax[3].plot(t, signal_filtered, color='r', lw=1, label='Clean Signal Retrieved')
	ax[3].set_ylim([minsignal, maxsignal])
	ax[3].set_xlabel('t axis')
	ax[3].set_ylabel('Accn in Gal')
	ax[3].legend()
	
	plt.subplots_adjust(hspace=0.6)
	plt.savefig('real-signal-analysis.png', bbox_inches='tight', dpi=300)	
	plt.show()
	
# ---------------------------ooo0ooo---------------------------
def plotRangeFFTs(tr):
    print('plotting FFT....')
    print(tr.stats)

    dt = tr.stats.delta
    Fs = 1 / dt  # sampling frequency
    tracestart = tr.stats.starttime
    #print(tracestart)
    #startSec =tracestart.seconds
    deltaTime=(tr.stats.endtime-tr.stats.starttime)
    print(deltaTime)

    t = np.arange(0, deltaTime, dt)  # create np array for time axis
    sigTemp = tr.data
    s = sigTemp[0:len(t)]

    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(7, 7))

    # plot time signal:
    axes[0, 0].set_title("Signal")
    axes[0, 0].plot(t, s, color='C0')
    axes[0, 0].set_xlabel("Time - s")
    axes[0, 0].set_ylabel("Amplitude - Pa")

    # plot different spectrum types:
    axes[1, 0].set_title("Magnitude Spectrum")
    axes[1, 0].magnitude_spectrum(s, Fs=Fs, color='C1')

    axes[1, 1].set_title("Log. Magnitude Spectrum")
    axes[1, 1].magnitude_spectrum(s, Fs=Fs, scale='dB', color='C1')

    axes[2, 0].set_title("Phase Spectrum ")
    axes[2, 0].phase_spectrum(s, Fs=Fs, color='C2')

    axes[2, 1].set_title("Power Spectrum Density")
    axes[2, 1].psd(s, 256, Fs, Fc=1)

    axes[0, 1].remove()  # don't display empty ax

    fig.tight_layout()
    plt.show()
    return


# ---------------------------ooo0ooo---------------------------

def openFolder():
    root = Tk()
    root.withdraw()
    #folder_selected = filedialog.askdirectory()
    folder_selected = filedialog.askdirectory(initialdir=os.getcwd(), title="Select directory")
    return folder_selected

# ---------------------------ooo0ooo---------------------------
def readInFolder(resamplefreq):
	z=openFolder()
	st = Stream()
	# Read in all files within slected directory.
	listing = os.listdir(z)
	for file in listing:
		if '.mseed' in file:
			print(file)
			streamTemp=obspy.read(z+'/'+file)

			streamTemp.resample(resamplefreq) 
			st += streamTemp
	st.sort(['starttime'])
	st.merge(method=1, fill_value="interpolate")
	return st



# ---------------------------ooo0ooo---------------------------
def simpleCorrelateTraces(tr1,tr2, lowCut, highCut):
	
	sumTraces=tr1.copy()
	#plotTwo(tr1, tr2, lowCut, highCut)
	tr1.data=np.sqrt(tr1.data**2)
	tr2.data=np.sqrt(tr2.data**2)

	if len(tr1.data) <= len(tr2.data):
		n=len(tr1.data)
	else:
		n=len(tr2.data)
	
	i=0
	while i < n:
		sumTraces.data[i]=tr1.data[i]+tr2.data[i]
		i=i+1
	sumTraces.data=sumTraces.data**2
	return sumTraces
#---------------------------ooo0ooo---------------------------
def save_as_mseed(st):
	start_time= st[0].stats.starttime
	year = str(start_time.year)
	month = str(start_time.month)
	day = str(start_time.day)
	hour = str(start_time.hour)
	
	
	filename = day + '_' + month +'_' + year  + '.mseed'
	st.write(filename, format="MSEED")

	print ('Data write completed')
	
	return None

# ---------------------------ooo0ooo---------------------------
def butter_bandpass(lowcut, highCut, samplingFreq, order=5):
    nyq = 0.5 * samplingFreq
    low = lowcut / nyq
    high = highCut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# ---------------------------ooo0ooo---------------------------

def butter_bandpass_filter(data, lowcut, highCut, samplingFreq, order=5):
    b, a = butter_bandpass(lowcut, highCut, samplingFreq, order=order)
    y = lfilter(b, a, data)
    return y

# ---------------------------ooo0ooo---------------------------
def CalcRunningMeanPower(tr, deltaT):
    N = len(tr)
    dt = tr.stats.delta
    newStream = tr.copy()
    x = newStream.data

    x = x ** 2

    nSamplePoints = int(deltaT / dt)
    runningMean = np.zeros((N - nSamplePoints), np.float32)

    # determie first tranche
    tempSum = 0.0

    for i in range(0, (nSamplePoints - 1)):
        tempSum = tempSum + x[i]

        runningMean[i] = tempSum

    # calc rest of the sums by subracting first value and adding new one from far end
    for i in range(1, (N - (nSamplePoints + 1))):
        tempSum = tempSum - x[i - 1] + x[i + nSamplePoints]
        runningMean[i] = tempSum
    # calc averaged acoustic intensity as P^2/(density*c)
    density_times_c = (1.2 * 330)
    runningMean=runningMean/nSamplePoints
    runningMean = runningMean / (density_times_c)

    newStream.data = runningMean
    newStream.stats.npts = len(runningMean)

    return newStream

    #---------------------------ooo0ooo---------------------------
def plot_daily_mag_field(tr):
    print(tr.stats)

    dt = tr.stats.delta
    n_data_points = len(tr.data)
    Fs = 1 / dt  # sampling frequency
    daily_start_datetime = tr.stats.starttime
    sample_end_time = tr.stats.endtime
    #print(tracestart)
    #startSec =tracestart.seconds
    deltaTime=(tr.stats.endtime-tr.stats.starttime)
    print(deltaTime)
    if (n_data_points > 20):
        ydata=tr.data[3:n_data_points]
        

        data=remove_glitches(ydata)
        # data=median_filter(ydata, len(ydata))

        zeroPoint = np.mean(ydata)
        data=ydata-zeroPoint #(de-mean the data)
        
        
        yearNow = daily_start_datetime.year
        monthNow = daily_start_datetime.month
        dayNow = daily_start_datetime.day
        
        dateString = str(yearNow) + '-' + str(monthNow) + '-' + str(dayNow)
        filename1 = dateString + '.svg'

        YAxisRange=np.amax(data)*1.1
        
        startHour=daily_start_datetime.hour
        startMinute=daily_start_datetime.minute
        timeSampled = sample_end_time-daily_start_datetime  #length of data in seconds
        graphStart = float(startHour) + float(startMinute/60)
        graphEnd = graphStart + (timeSampled/3600.0)
        xValues= np.linspace(graphStart, graphEnd, len(ydata))
        
        z = UTCDateTime()
        upDated= z.strftime("%A, %d. %B %Y %I:%M%p")
            
        fig = plt.figure(figsize=(12,4))
        xAxisText="Time  updated - "+ upDated +" UTC"
        plt.title('B-Field E-W, Guisborough, UK - '+ dateString)
        plt.xlabel(xAxisText)
        zeroLabel=str('%0.5g' % (zeroPoint))
        plt.ylabel('$\Delta$Flux Density - nT     0.0='+zeroLabel+'nT')
        plt.plot(xValues, data,  marker='None',    color = 'darkolivegreen')  
        plt.xlim(0, 24.1)
        plt.xticks(np.arange(0, 24.1, 3.0))
        plt.grid(True)
        plt.savefig(filename1)
        plt.show()
     
        plt.close('all')     


    return

def remove_glitches(ydata):
  #to maintain continuity of data-line miss-reads are assigned B=0.0
  z=len(ydata)
  dataNew=ydata

  for i in range (0,z-4):
    if (abs(ydata[i]) < 0.1):
        dataChunk=[ydata[i],ydata[i+1],ydata[i+2],ydata[i+3]]
        dataNew[i]=statistics.median(dataChunk)
  
  dataNew[(z-3):(z-1)]=ydata[(z-3):(z-1)]
  return dataNew
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$


#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

