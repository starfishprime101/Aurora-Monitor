# ---------------------------ooo0ooo---------------------------
#       Seismic Monitoring Software
#       Ian Robinson
#       http://schoolphysicsprojects.org
#
#
#       requires
#           python3, python3-obspy, matplotlib
#           icp10125- https://github.com/pimoroni/icp10125-python
# ---------------------------Notes---------------------------
#
#   ## loose weekly readings
#
# ---------------------------ooo0ooo---------------------------

import time
import smbus
import time
import serial
from threading import Thread
from geostationModules import *
from obspy.signal.filter import bandpass, lowpass, highpass

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# Global Variables (actually constants)

#device = ICP10125()

os.chdir(home_directory)

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
# -----create top level Data and plots directories if not already present
# ------------------------------------------------------#
#                                                       #
#                       Main Body                       #
#                                                       #
# ------------------------------------------------------#
def main():
    # -----create top level Data and plots directories if not already present

    here = os.path.dirname(os.path.realpath(__file__))

    try:
        os.makedirs('Plots/')
    except OSError:
        if not os.path.isdir('Plots/'):
            raise

    try:
        os.makedirs('Data/')
    except OSError:
        if not os.path.isdir('Data/'):
            raise

# ---------------------------ooo0ooo---------------------------
    # specifically for reading value from Arduinno
    ser = serial.Serial('/dev/ttyACM0', 57600)
    ser.flushInput()
    time.sleep(2)

    ###*******FGM-3 Sensor Values**********
    mSens = 0.000609756
    cSens = -41.524
    ###************************************

    # ---------------------------ooo0ooo---------------------------
    first_week = True  # queue behaviour different when queue not yet fully filled

    # two circular queues are used to store long-term data (>1 day) for plotting
    # queuePrev168hrs stores data points in a numpy array
    # to avoid timing drift q_index_hourly_starttimes stores the start times and pointers to
    #           the tail position of each hour's data in queuePrev168hrs

    # -- create numpy arrays to store pressure and time data
    tranche_prev_minute = np.zeros([N_TARGET_PREV_MINUTE, 2], dtype=np.float32)
    # implemented as circular queue
    q_prev_168hrs_data = np.zeros([N_TARGET_WEEKLY_SAMPLES, 2], np.float32)
    daily_readings = np.zeros([N_TARGET_DAILY_SAMPLES, 2], np.float32)
    hourly_readings = np.zeros([N_TARGET_HOURLY_SAMPLES, 2], np.float32)
    weekly_readings = np.zeros([N_TARGET_WEEKLY_SAMPLES, 2], dtype=np.float32)

    # create list to hold tail pointers and hourly start times of previous week's data
    q_index_hourly_starttimes = []

    n_prev_minute = 0
    n_daily_readings = 0
    n_hourly_readings = 0
    n_weekly_readings = 0
    hp_prev_168hrs_data = 0
    tp_prev_168hrs_data = 0
    tp_index_hourly_starttimes = 0
    hp_index_hourly_starttimes = 0

    # Initialise queue for the hourly startimes and data positions in data queue
    # tmp=[UTCDateTime(), 0] #0 will be replaced by the head pointer, i.e. start position of that hour's readings
    for x in range(168):
        tmp = [UTCDateTime(), x]
        q_index_hourly_starttimes.append(tmp)

    time.sleep(30)  # wait 30 sec before starting

    #discard first 3 readings to avoid start-up glitches
    try:
        tmp_readings = read_bfield_from_sensor(ser, mSens, cSens)
        print(tmp_readings )
        time.sleep(5)
        tmp_readings = read_bfield_from_sensor(ser, mSens, cSens)
        print(tmp_readings )
        time.sleep(5)
        print(tmp_readings )
        time.sleep(5)
    except IOError:
        print('read error at first readings')

    week_start = UTCDateTime()
    lastSave = week_start
    day_start = week_start  # initialise variable
    hour_start = week_start
    minute_start = week_start
    end_time_prev168hrs_data = week_start  # initialise variable


    while 1:

        time.sleep(SAMPLING_PERIOD)

        try:
            tmp_readings = read_bfield_from_sensor(ser, mSens, cSens)
            field = tmp_readings [0]
            xxx = tmp_readings [1]
            daily_readings[n_daily_readings, 0] = field
            hourly_readings[n_hourly_readings, 0] = field
            tranche_prev_minute[n_prev_minute, 0] = field
            daily_readings[n_daily_readings, 1] = xxx
            hourly_readings[n_hourly_readings, 1] = xxx
            tranche_prev_minute[n_prev_minute, 1] = xxx
            n_daily_readings = n_daily_readings + 1
            n_hourly_readings = n_hourly_readings + 1
            n_prev_minute = n_prev_minute + 1

        except IOError:
            print('read error')
            daily_readings[n_daily_readings, 0] = 0.00
            daily_readings[n_daily_readings, 1] = 0.00
            hourly_readings[n_hourly_readings, 0] = 0.00
            hourly_readings[n_hourly_readings, 1] = 0.00
            n_daily_readings = n_daily_readings + 1
            n_hourly_readings = n_hourly_readings + 1

        # ------------------------------------------------
#       calc average of last minute's values storing in weekly_readings
        if UTCDateTime().minute != minute_start.minute:
            if n_prev_minute > 0:
                mean_pressure = np.average(
                    tranche_prev_minute[0:n_prev_minute, 0])
                mean_temperature = np.average(
                    tranche_prev_minute[0:n_prev_minute, 1])
                weekly_readings[n_weekly_readings, 0] = mean_pressure
                weekly_readings[n_weekly_readings, 1] = mean_temperature

                q_prev_168hrs_data[hp_prev_168hrs_data, 0] = mean_pressure
                q_prev_168hrs_data[hp_prev_168hrs_data, 1] = mean_temperature

                # add 1 to hp wrapping around to 0 if > N_TARGET_WEEKLY_SAMPLES % = 'mod'
                hp_prev_168hrs_data = (hp_prev_168hrs_data +
                                       1) % N_TARGET_WEEKLY_SAMPLES
                # reset tranche to all zeros
                tranche_prev_minute = np.zeros(
                    [N_TARGET_PREV_MINUTE, 2], dtype=np.float32)
                n_weekly_readings = n_weekly_readings + 1
                minute_start = UTCDateTime()
                n_prev_minute = 0

        # ------------------------------------------------
        if UTCDateTime().hour != hour_start.hour:
            end_time_prev168hrs_data = UTCDateTime()
            prev_hour_start = hour_start
            hour_start = UTCDateTime()

            if hp_index_hourly_starttimes == 167:  # check whether full week of data has been gathered
                first_week = False

            # wrap around to beginning at 168 hrs
            hp_index_hourly_starttimes = (hp_index_hourly_starttimes + 1) % 168
            # for 1st 168hrs tailpointer=0 .. 1st wrap hp..168->0 and tp -> 1, after that
            # increment tp by 1 each time
            if first_week is False:
                tp_index_hourly_starttimes = (
                    tp_index_hourly_starttimes + 1) % 168

            # write new hour's start position and timestamp to circular queue
            q_index_hourly_starttimes[hp_index_hourly_starttimes][0] = hour_start
            q_index_hourly_starttimes[hp_index_hourly_starttimes][1] = hp_prev_168hrs_data

            tmp_prev_168hr_data = np.zeros(
                [N_TARGET_WEEKLY_SAMPLES, 2], np.float32)

            # look up pointer to position for first data item
            tp_prev_168hrs_data = q_index_hourly_starttimes[tp_index_hourly_starttimes][1]
            i = tp_prev_168hrs_data
            ntmp_prev_168hr_data = 0

            while i != hp_prev_168hrs_data:
                tmp_prev_168hr_data[ntmp_prev_168hr_data,
                                    0] = q_prev_168hrs_data[i, 0]
                tmp_prev_168hr_data[ntmp_prev_168hr_data,
                                    1] = q_prev_168hrs_data[i, 1]
                ntmp_prev_168hr_data = ntmp_prev_168hr_data + 1
                i = (i + 1) % N_TARGET_WEEKLY_SAMPLES  # wrap at end of queue

            # start plotting as thread to hopefully reduce chance of 'glitching' on sensor reads.
            if ntmp_prev_168hr_data > 10:
                thread_plot_and_save = Thread(target=save_and_plot_all, args=((
                    hourly_readings, n_hourly_readings, prev_hour_start,
                    daily_readings, n_daily_readings, day_start,
                    weekly_readings, n_weekly_readings, week_start,
                    tmp_prev_168hr_data, ntmp_prev_168hr_data, end_time_prev168hrs_data,
                    q_index_hourly_starttimes, tp_index_hourly_starttimes)))

                thread_plot_and_save.start()

            # reset data array to zero for new hour
            hourly_readings = np.zeros(
                [N_TARGET_HOURLY_SAMPLES, 2], np.float32)
            n_hourly_readings = 0
        # ------------------------------------------------
        if day_start.day != UTCDateTime().day:

            # reset data array to zero for new day
            daily_readings = np.zeros([N_TARGET_DAILY_SAMPLES, 2], np.float32)
            day_start = UTCDateTime()
            n_daily_readings = 0

            if end_time_prev168hrs_data.isoweekday() == 1:  # Monday-- new week starts
                # clear theweekly data array
                weekly_readings = np.zeros(
                    [N_TARGET_WEEKLY_SAMPLES, 2], dtype=np.float32)
                week_start = UTCDateTime()
                n_weekly_readings = 0


# ---------------------------ooo0ooo---------------------------
def read_from_sensor():
    # this function specifically reads data over i2c from a
    # icp10125
    pressure, temperature = device.measure()
    readings = (pressure, temperature)

    return readings

# ---------------------------ooo0ooo---------------------------
def read_bfield_from_sensor(ser, mSens, cSens):
    try:
        freq = np.float32(ser.readline())   # converts the freq number string to a floating point number
        if (freq > 10000):
            b_field = ((mSens*freq)+cSens)*1000.0# convert freq to field in nTeslas
        else:
            b_field = 0.0
        readings = (b_field, 0.0)
    except ValueError:
        f.write ('failed to read serial line correctly --trying again')
    return readings
# ---------------------------ooo0ooo---------------------------

def save_and_plot_all(hourly_readings, n_hourly_readings, prev_hour_start,
                      daily_readings, n_daily_readings, day_start,
                      weekly_readings, n_weekly_readings, week_start,
                      tmp_prev_168hr_data, ntmp_prev_168hr_data, end_time_prev168hrs_data,
                      q_index_hourly_starttimes, tp_index_hourly_starttimes):

    channel_no = 0
    st = create_mseed(weekly_readings, week_start,
                      end_time_prev168hrs_data, n_weekly_readings, channel_no)
    save_weekly_data_as_mseed(st, channel_no)
    plot_weekly(st, channel_no)

    st = create_mseed(hourly_readings, prev_hour_start,
                      end_time_prev168hrs_data, n_hourly_readings, channel_no)
    save_hourly_data_as_mseed(st, channel_no)

    # create daily stream and plot
    st = create_mseed(daily_readings, day_start,
                      end_time_prev168hrs_data, n_daily_readings, channel_no)
    plot_daily(st, channel_no)

    if ntmp_prev_168hr_data > 10:
        start_time_prev168hrs_data = q_index_hourly_starttimes[tp_index_hourly_starttimes][0]
        plotPrev168hrs(tmp_prev_168hr_data, start_time_prev168hrs_data,
                       end_time_prev168hrs_data, ntmp_prev_168hr_data, channel_no)

    if MULTIPLE_SENSORS:
        channel_no = 1
        st = create_mseed(weekly_readings, week_start,
                          end_time_prev168hrs_data, n_weekly_readings, channel_no)
        save_weekly_data_as_mseed(st, channel_no)
        plot_weekly(st, channel_no)

        st = create_mseed(hourly_readings, prev_hour_start,
                          end_time_prev168hrs_data, n_hourly_readings, channel_no)
        save_hourly_data_as_mseed(st, channel_no)

        # create daily stream and plot
        st = create_mseed(daily_readings, day_start,
                          end_time_prev168hrs_data, n_daily_readings, channel_no)
        plot_daily(st, channel_no)

        if ntmp_prev_168hr_data > 10:
            start_time_prev168hrs_data = q_index_hourly_starttimes[tp_index_hourly_starttimes][0]
            plotPrev168hrs(tmp_prev_168hr_data, start_time_prev168hrs_data,
                           end_time_prev168hrs_data, ntmp_prev_168hr_data, channel_no)


# ---------------------------ooo0ooo---------------------------

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#  runs topmost 'main' function
if __name__ == '__main__':
    main()
# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
