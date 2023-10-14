#---------------------------ooo0ooo---------------------------
#       Infrasound Monitoring Software
#       Ian Robinson
#       http://schoolphysicsprojects.org
#
#
#		requires
#			python3, python3-obspy, matplotlib
#			icp10125- https://github.com/pimoroni/icp10125-python
#---------------------------Notes---------------------------
#
#		last update 26/11/22
#
#---------------------------ooo0ooo---------------------------

	
# ---------------------------ooo0ooo---------------------------
# import required modules
import numpy as np
import time
import array
import multiprocessing
import obspy

from seismic_analysis_routines import *
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
#             			Main Body           			#
#                                  						#
# ------------------------------------------------------#

def main():
	resamplefreq=1.0 #used when joining multiple data files -- should be a few Hz below the actual sample rate
	lowCut = 1.0/80000	# low frequency cut-off
	highCut = 1.0/60 # high frequency cut-off - using just under half the signal sample rate
	# deltaT = 0.1  	# time interval seconds to calculate running mean for acoustic power

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	###~~~read in single datafile or an entire folder
	## select on one of the two options below
	
	st1 = readDataFile()  # select a data file to work on

	#st1=readDataFolder(resamplefreq)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

	print(st1[0].stats)
	print("#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
	
	tr1 = st1[0].copy()  # extract the data 'trace''
	tr1.detrend(type='simple') # effectivly zeroes the trace on the mean value


	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# #this allows a slice of the trace to be selected
	# # ### ----- select slice of data to work on
	# startMinute =  30           #edit this value
	# endMinute =  58            #edit this value
	
	# tracestart = tr1.stats.starttime
	# startSec = (startMinute * 60)
	# endSec = (endMinute * 60)
	# tr1.trim(tracestart + startSec, tracestart + endSec)
	# # #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	#  one should play with the best values of lowCut and highCut below
	tr1.filter('bandpass', freqmin=lowCut, freqmax=highCut, corners=4, zerophase=True)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# plotSpectrogram1(tr1)
	simplePlot(tr1)
	plot_daily_mag_field(tr1)
	# plotDayplot(tr1, lowCut, highCut)

	# plotBands(tr1,deltaT)
	#plotAcousticPower(tr1, deltaT)

	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# note: wavelet transform is computationally demanding and takes time
	#typically limit to 1hr of data or less
	# plotWaveletTransformBeta(tr1, lowCut, highCut)
	#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	
	
	# plotWelchPeriodogram(tr1, lowCut, highCut)
	#plotSpectrogram1(tr1)
	# plotSpectrogram2(tr1,lowCut, highCut)
	# plotSpectrogram3(tr1,lowCut, highCut)
	# plotSpectrogram4(tr1)
	# ~ plotPwrMagnitudeSpectrumLin(tr1)
	#plotPwrMagnitudeSpectrumLog(tr1)
	# ~ plotRangeFFTs(tr1)
	

	

	
	#plotDayPlotAcousticPower(tr1, deltaT, lowCut, highCut)
	
	
	#~~~~~~~~~~~~~~~~~~~experimental~~~~~~~~~~~~~~~~~~~~~~
	#plotSimpleFFT(tr1)
	#plotFFT(tr1)
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~




#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  runs topmost 'main' finction
if __name__ == '__main__':
    main()
#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
