
# CODE:
#
# Note: this file is taken from:
#     https://github.com/guillaume-chevalier/HAR-stacked-residual-bidir-LSTMs
#     (Apache 2.0 License, Copyright 2017, Guillaume Chevalier and Yu Zhao)
# Which is in turns derived from:
#     https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition/blob/master/data/download_dataset.py
#     (MIT License, Copyright 2017, Guillaume Chevalier)

# DATASETS:
#
# The first dataset downloaded is this one (UCI HAR dataset):
#     Davide Anguita, Alessandro Ghio, Luca Oneto, Xavier Parra and Jorge L. Reyes-Ortiz. A Public Domain Dataset for Human Activity Recognition Using Smartphones. 21th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, ESANN 2013. Bruges, Belgium 24-26 April 2013.
#     https://archive.ics.uci.edu/ml/datasets/Human+Activity+Recognition+Using+Smartphones
#         !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"
#         !wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.names"
# The second dataset downloaded is this one (Opportunity dataset):
#     Daniel Roggen, Alberto Calatroni, Mirco Rossi, Thomas Holleczek, Gerhard Tröster, Paul Lukowicz, Gerald Pirkl, David Bannach, Alois Ferscha, Jakob Doppler, Clemens Holzmann, Marc Kurz, Gerald Holl, Ricardo Chavarriaga, Hesam Sagha, Hamidreza Bayati, and José del R. Millàn. "Collecting complex activity data sets in highly rich networked sensor environments" In Seventh International Conference on Networked Sensing Systems (INSS’10), Kassel, Germany, 2010.
#     http://www.opportunity-project.eu/challengeDataset
#         !wget https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip
#


import os
from subprocess import call


print("")

print("Downloading UCI HAR Dataset...")
if not os.path.exists("UCI HAR Dataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI HAR Dataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
extract_directory = os.path.abspath("UCI HAR Dataset")
if not os.path.exists(extract_directory):
    call(
        'unzip -nq "UCI HAR Dataset.zip"',
        shell=True
    )
    print("Extracting successfully done to {}.".format(extract_directory))
else:
    print("Dataset already extracted. Did not extract twice.\n")


print("Downloading opportunity dataset...")
if not os.path.exists("OpportunityUCIDataset.zip"):
    call(
        'wget "https://archive.ics.uci.edu/ml/machine-learning-databases/00226/OpportunityUCIDataset.zip"',
        shell=True
    )
    print("Downloading done.\n")
else:
    print("Dataset already downloaded. Did not download twice.\n")


print("Extracting...")
if not os.path.exists("oppChallenge_gestures.data"):
    from preprocess_data import generate_data
    generate_data("OpportunityUCIDataset.zip", "oppChallenge_gestures.data", "gestures")
    print("Extracting successfully done to oppChallenge_gestures.data.")
else:
    print("Dataset already extracted. Did not extract twice.\n")
