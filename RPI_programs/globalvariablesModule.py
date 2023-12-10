# global Constants

import os

home_directory = '/home/geophysics/station'
# replace with home directory of application

# below are station parameters for your station. see SEED manual for
# more details http://www.fdsn.org/pdf/SEEDManual_V2.4.pdf  esp. Apprendix A

MULTIPLE_SENSORS = False   # only channel 0 will be used if false
# -- station parameters
# change this to your sensor # channel M=mid-period 1-10Hz sampling, D=pressure sensor, O = outside
STATION_ID = 'xxxx'
# change this to your sensor id # channel M=mid-period 1-10Hz sampling, K=temperature sensor, O = outside

STATION_CHANNEL_0 = 'LFE'  # see SEED format documentation
STATION_CHANNEL_1 = 'xxx'  # see SEED format documentation
STATION_LOCATION = '01'  # 2 digit code to identify specific sensor rig
STATION_NETWORK = 'IR'
STATION_INFO_0 = STATION_ID + '-' + STATION_CHANNEL_0 + '-' + STATION_LOCATION
STATION_INFO_1 = STATION_ID + '-' + STATION_CHANNEL_1 + '-' + STATION_LOCATION

# filesnames for plots =
DATATYPE_CHANNEL_0 = 'B_Field'
PLOT_TITLE_CHANNEL_0 = 'B Field, E-W,  Guisborough, UK '
PLOT_YLABEL_CHANNEL_0 = 'Flux Density - nT  :  '

DATATYPE_CHANNEL_1 = 'xxx'
PLOT_TITLE_CHANNEL_1 = 'xxx'
PLOT_YLABEL_CHANNEL_1 = 'xxx'

SAMPLING_FRQ = 0.5
SAMPLING_PERIOD = 1.00/SAMPLING_FRQ
AVERAGINGTIME = 30  # time interval seconds to calculate running mean

N_TARGET_HOURLY_SAMPLES = int(SAMPLING_FRQ*3600*1.5)
N_TARGET_DAILY_SAMPLES = int(N_TARGET_HOURLY_SAMPLES * 24)
N_TARGET_PREV_MINUTE = int(SAMPLING_FRQ*60*1.3)

#no of weekly samples depends on averaging period e.g. 1 per minute ~10080
N_TARGET_WEEKLY_SAMPLES = 12000
