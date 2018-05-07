
# Note: this file is taken from:
#     https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
#     (Apache 2.0 License, Copyright 2017, Guillaume Chevalier and Yu Zhao)
# Which is in turns derived from there:
#     https://github.com/guillaume-chevalier/python-signal-filtering-stft
#     (MIT License, Copyright 2017, Guillaume Chevalier)

# Note: see explanations here about what this code does:
#     https://github.com/guillaume-chevalier/python-signal-filtering-stft
# Our signal is sampled and processed slightly differently here. Hence,
# see the function `filter_opportunity_datasets_accelerometers` below.


__author__ = 'Guillaume Chevalier'


import numpy as np
from scipy import signal


def butter_lowpass(cutoff, nyq_freq, order=4):
    # Build the filter
    normal_cutoff = float(cutoff) / nyq_freq
    b, a = signal.butter(order, normal_cutoff, btype='lowpass')
    return b, a

def butter_lowpass_filter(data, cutoff_freq, nyq_freq, order=4):
    # Build and apply filter to data (signal)
    b, a = butter_lowpass(cutoff_freq, nyq_freq, order=order)
    y = signal.filtfilt(b, a, data)
    return y

def filter_opportunity_datasets_accelerometers(accelerometer_data):
    # Cutoff frequencies and filters inspired from:
    # https://www.elen.ucl.ac.be/Proceedings/esann/esannpdf/es2013-84.pdf
    # Their butterworth filter's order is of 3.

    # Note: here we have a 30 Hz sampling rate on the opportunity dataset,
    # which is diffrerent than in the paper above.
    opportunity_dataset_sampling_rate = 30.0
    nyq_freq = opportunity_dataset_sampling_rate / 2.0

    new_channels = []
    for channel in accelerometer_data.transpose():
        # LP filter to 0.3 Hz for splitting gravity component from body.
        gravity = butter_lowpass_filter(channel, 0.3, nyq_freq, order=3)

        body = channel
        # We assume that body acc has lost its gravity componenent
        body -= gravity

        new_channels.append(body)
        new_channels.append(gravity)

    preprocessed_data = np.array(new_channels).transpose()
    return preprocessed_data
