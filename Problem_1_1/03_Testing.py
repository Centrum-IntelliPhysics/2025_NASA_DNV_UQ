# Copyright (C) 2025 Sukanta Basu
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
File: Preprocess.py
===========================

:Author: Sukanta Basu
:Date: 2025-3-27
:Description: perform training using FLAML
"""

# ============================================
# Imports
# ============================================

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import os
import pickle
from pickle import dump, load
import time

from flaml import AutoML

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# For reproducibility of the results, the following seeds should be selected
from numpy.random import seed
seed(30)
randSeed = np.random.randint(1000)


# ============================================
# User input
# ============================================

SIM_STR = 'Xc_050_850_850/'
N_STR = '1M/'

# Number of ensembles
nEns = 25

# Tuning time for FLAML
maxTime = 600  # 24*3600


# ============================================
# Input & output directories
# ============================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_DNV_UQ/Version03/'
# ROOT_DIR = '/Users/sukantabasu/Dropbox/Priority/Works_AIML/2025_DNV_UQ/Version03/'

ExtractedDATA_DIR = ROOT_DIR + 'DATA/ExtractedDNVDATA/' + SIM_STR
Tuning_DIR = ROOT_DIR + 'Tuning/FLAML/' + SIM_STR + N_STR
FIG_DIR = ROOT_DIR + 'Figures/' + SIM_STR + N_STR

nSample = 100

for param in range(1,5+1,1):

    if param < 5:

        XDNV = np.load(ExtractedDATA_DIR + 'XDNV_Base.npy')

    else:  # contains aleatoric and epistemic parameters as input

        XDNV = np.load(ExtractedDATA_DIR + 'XDNV_Base.npy')
        X_a0 = np.load(ExtractedDATA_DIR + 'XDNV_Param_1_p50.npy')
        X_a1 = np.load(ExtractedDATA_DIR + 'XDNV_Param_2_p50.npy')
        X_e0 = np.load(ExtractedDATA_DIR + 'XDNV_Param_3_p50.npy')
        X_e1 = np.load(ExtractedDATA_DIR + 'XDNV_Param_4_p50.npy')
        XDNV = np.hstack([XDNV,X_a0,X_a1,X_e0,X_e1])

    YDNV_pred_combo = np.zeros((nSample, nEns))

    for n in range(nEns):
        fSTR = (Tuning_DIR + 'FLAML_' + 'param' + str(param) + '_maxT' + str(maxTime) + '_ens' + str(n) + '.pkl')

        with open(fSTR, "rb") as f:
            automl = pickle.load(f)

        # Make Prediction for blind set
        YDNV_pred_combo[:, n] = automl.predict(XDNV)

    YDNV_pred_p50 = np.median(YDNV_pred_combo, axis=1)
    YDNV_pred_full = YDNV_pred_combo.ravel()  # Size 100 x NEns

    YDNV_pred_p50 = YDNV_pred_p50.reshape(-1, 1)
    YDNV_pred_p50 = np.clip(YDNV_pred_p50, 0, 1)  # Clipping
    np.save(ExtractedDATA_DIR + 'XDNV_Param_' + str(param) + '_p50', YDNV_pred_p50)

    YDNV_pred_full = np.clip(YDNV_pred_full, 0, 1)  # Clipping
    np.save(ExtractedDATA_DIR + 'XDNV_Param_' + str(param), YDNV_pred_full)