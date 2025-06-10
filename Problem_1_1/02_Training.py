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
:Date: 2025-3-24
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

from sklearn.utils import shuffle

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error
)
from scipy.stats import pearsonr

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

SIM_STR = 'Xc_500_850_850/'
N_STR = '1M/'

# Number of ensembles
nEns = 25 

# Tuning time for FLAML
maxTime = 600  # 24*3600

# Train-Val split ratio
splitRatio = 0.2

# Decision tree approaches
model = ['lgbm']


# ============================================
# Input & output directories
# ============================================

ROOT_DIR = '/data/Sukanta/Works_AIML/2025_DNV_UQ/Version03/'
# ROOT_DIR = '/Users/sukantabasu/Dropbox/Priority/Works_AIML/2025_DNV_UQ/Version03/'

DATA_DIR = ROOT_DIR + 'DATA/SimulatedDATA/' + SIM_STR + N_STR
ExtractedDATA_DIR = ROOT_DIR + 'DATA/ExtractedDATA/' + SIM_STR + N_STR
Tuning_DIR = ROOT_DIR + 'Tuning/FLAML/' + SIM_STR + N_STR
FIG_DIR = ROOT_DIR + 'Figures/' + SIM_STR + N_STR

nSample = 1000000
nTest = int(0.1*nSample)

statsFile = os.path.join(Tuning_DIR, f'Summary_Stats_maxTime_{maxTime}.txt')

# Optional: start fresh by removing the file if it exists
if os.path.exists(statsFile):
    os.remove(statsFile)

for param in range(1,5+1,1):

    if param < 5:

        XFull = np.load(ExtractedDATA_DIR + 'XTrnVal_Base.npy')

    else:  # contains aleatoric and epistemic parameters as input

        XFull = np.load(ExtractedDATA_DIR + 'XTrnVal_Step4.npy')

    YFull = np.load(ExtractedDATA_DIR + 'YTrnVal.npy')

    YFull = YFull[:, param - 1: param]

    # Use the last 100k sample for testing
    XTst = XFull[nSample - nTest:, :]
    YTst = YFull[nSample - nTest:, :]

    # Use the first 900k sample for train + validation
    XTrnVal = XFull[:nSample - nTest, :]
    YTrnVal = YFull[:nSample - nTest, :]

    print(np.shape(XTrnVal))
    print(np.shape(YTrnVal))

    automl = AutoML()
    automl_settings = {
        "time_budget": maxTime,  # in seconds
        "metric": 'mse',
        "task": 'regression',
        "estimator_list": model,
        "eval_method": 'holdout',
        "n_splits": 5,
        "n_jobs": 12, # only use on HAL
        "early_stop": True,
        "sample": True,  # A boolean of whether to sample the training data during search
        "ensemble": 0,  # Whether to perform ensemble after search
        "model_history": True,  # A boolean of whether to keep the best model per estimator
        "retrain_full": True  # whether to retrain the selected model on the full training data
    }

    nTrnVal, nFeatures = np.shape(XTrnVal)
    nVal = np.rint(nTrnVal * splitRatio).astype(int)

    YTst_pred_ens = np.zeros((nTest, nEns))

    for n in range(nEns):

        # Shuffle data
        r = np.random.randint(1000)
        XTrnVal, YTrnVal = shuffle(XTrnVal, YTrnVal, random_state=r)

        # Split competition data into train and validation
        XTrn = XTrnVal[nVal:, :]
        YTrn = YTrnVal[nVal:, :]

        XVal = XTrnVal[0:nVal, :]
        YVal = YTrnVal[0:nVal, :]

        automl.fit(X_train=XTrn, y_train=YTrn.squeeze(), X_val=XVal, y_val=YVal.squeeze(), **automl_settings)

        print(automl.best_config_per_estimator)
        print('Best ML learner:', automl.best_estimator)
        print('Best hyperparmeter config:', automl.best_config)
        print('Best accuracy on validation data: {0:.4g}'.format(automl.best_loss))
        print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

        fSTR = (Tuning_DIR + 'FLAML' + '_param' + str(param) + '_maxT' + str(maxTime) + '_ens' + str(n) + '.pkl')
        with open(fSTR, "wb") as f:
            dump(automl, f, pickle.HIGHEST_PROTOCOL)

        # Make prediction for test set
        YTst_pred = automl.predict(XTst)
        YTst_pred = np.clip(YTst_pred, 0, 1)  # Clipping
        mse = np.mean((np.squeeze(YTst) - np.squeeze(YTst_pred)) ** 2)

        YTst_pred_ens[:, n] = YTst_pred

    YTst_pred_p50 = np.median(YTst_pred_ens, axis=1)

    # Compute error metrics
    y_true = YTst.squeeze()
    y_pred = YTst_pred_p50.squeeze()

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    pearson_r, _ = pearsonr(y_true, y_pred)

    strStats = (
        f"param {param} | RMSE = {rmse:.3e} | MAE = {mae:.3e} | "
        f"R2 = {r2:.3e} | MAPE = {mape:.3e}% | PearsonR = {pearson_r:.3e}"
    )
    print(strStats)

    with open(statsFile, 'a') as f:
        f.write(strStats + '\n')

    plt.plot(np.squeeze(YTst), np.squeeze(YTst_pred_p50), '.k')
    plt.xlabel('Actual Input Parameter')
    plt.ylabel('Predicted Input Parameter')
    plt.savefig(
        FIG_DIR + 'Scatter_' + 'param_' + str(param) + '_maxTime_' + str(maxTime) + '.png',
        dpi=300, bbox_inches='tight')
    plt.close()

    plt.hexbin(np.squeeze(YTst), np.squeeze(YTst_pred_p50), gridsize=100)
    plt.plot([0, 1], [0, 1], 'r-', linewidth=2)

    plt.xlabel('Actual Input Parameter')
    plt.ylabel('Predicted Input Parameter')
    plt.savefig(
        FIG_DIR + 'Hexbin_' + 'param_' + str(param) + '_maxTime_' + str(maxTime) + '.png',
        dpi=300, bbox_inches='tight')
    plt.close()
