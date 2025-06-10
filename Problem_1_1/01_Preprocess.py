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
:Description: preprocess response series
"""

# ============================================
# Imports
# ============================================

import numpy as np
import pandas as pd

import scipy.signal
from scipy.stats import linregress
from statsmodels.tsa.stattools import acf
import antropy as ant
import nolds

import warnings
from sklearn.exceptions import UndefinedMetricWarning
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# ============================================
# User input
# ============================================

SIM_STR = 'Xc_950_550_950/'
N_STR = '1M/'
optTrnTst = 0 #1: Train, 0: Test


# ============================================
# Input & output directories
# ============================================

ROOT_DIR = '/Users/sukantabasu/Dropbox/Priority/Works_AIML/2025_DNV_UQ/Version03/'

if optTrnTst == 1:

    DATA_DIR = ROOT_DIR + 'DATA/SimulatedDATA/' + SIM_STR + N_STR
    ExtractedDATA_DIR = ROOT_DIR + 'DATA/ExtractedDATA/' + SIM_STR + N_STR
    nSample = 1000000

else:

    DATA_DIR = ROOT_DIR + 'DATA/DNVDATA/' + SIM_STR
    ExtractedDATA_DIR = ROOT_DIR + 'DATA/ExtractedDNVDATA/' + SIM_STR
    nSample = 100


# ============================================
# Convert raw simulation data
# ============================================

if optTrnTst == 0:

    df = pd.read_csv(DATA_DIR + 'Y_out_example.csv', skiprows=2, header=None)

    sample_indices = df[6].unique()  # Extract unique sample indices
    num_samples = len(sample_indices)
    df = df.drop(columns=[6])  # Drop the sample index column
    Y_out = df.to_numpy().reshape(num_samples, 60, 6).transpose(1, 2, 0)

    num_responses = 6
    for k in range(num_responses):
        y_response = np.zeros((num_samples, 60))
        for i in range(num_samples):
            y_response[i, :] = Y_out[:, k, i]

        filename = f'y_response_{k}.txt'
        np.savetxt(DATA_DIR + filename, y_response, fmt='%.4f', delimiter=' ')


# ============================================
# Function: central moment computation
# ============================================

def central_moment(y, order):
    mean_y = np.mean(y, axis=1, keepdims=True)
    return np.mean((y - mean_y) ** order)


# ============================================
# Function: spectral moment computation
# ============================================

def compute_spectral_moments(signal, fs=1.0):
    """
    Computes the zeroth, first, second, third, and fourth spectral moments.

    Parameters:
        signal : ndarray
            Input time series.
        fs : float, optional
            Sampling frequency (default=1.0).

    Returns:
        moments : dict
            Dictionary containing M0, M1, M2, M3, M4.
    """
    # Compute the Power Spectral Density (PSD)
    freqs, psd = scipy.signal.welch(signal, fs=fs, nperseg=len(signal))

    # Compute Spectral Moments
    M0 = np.sum(psd)  # Total power
    M1 = np.sum(freqs * psd) / M0  # Spectral centroid (mean frequency)
    M2 = np.sum((freqs - M1) ** 2 * psd) / M0  # Spectral variance
    M3 = np.sum((freqs - M1) ** 3 * psd) / (M0 * M2 ** (3 / 2))  # Spectral skewness
    M4 = np.sum((freqs - M1) ** 4 * psd) / (M0 * M2 ** 2)  # Spectral kurtosis

    return np.array([M0, M1, M2, M3, M4])


# ============================================
# Function: autocorrelation stats computation
# ============================================

# At what lag, the autocorrelation drops to 0.5, 1/e
def lag_autocor(y):
    y = y - np.mean(y)
    Corr = acf(y, nlags=np.round(y.size / 2))

    indx_p50 = np.squeeze(np.where(Corr <= 0.5))
    if indx_p50.size > 0:
        Lag_p50 = indx_p50[0]
    else:
        Lag_p50 = len(y)

    indx_e = np.squeeze(np.where(Corr <= 1 / np.exp(1)))
    if indx_e.size > 0:
        Lag_e = indx_e[0]
    else:
        Lag_e = len(y)

    return np.array([Lag_p50, Lag_e])


# ============================================
# Function: slopes & intercept computation
# ============================================

def compute_slope_intercept_scipy(y1, y2):
    m, c, _, _, _ = linregress(y1, y2)
    return m, c


# ============================================
# Compute statistics from 6 response series
# ============================================

centralMoments = np.zeros((nSample, 24))
centralMomentsInc = np.zeros((nSample, 24))
minMax = np.zeros((nSample, 12))
minMaxInc = np.zeros((nSample, 12))

cnt = 0
for i in range(0, 6, 1):

    y = np.loadtxt(DATA_DIR + 'y_response_' + str(i) + '.txt')
    dy = np.diff(y, axis=1)

    centralMoments[:, cnt] = np.mean(y, axis=1)
    centralMomentsInc[:, cnt] = np.mean(dy, axis=1)
    cnt = cnt + 1

    centralMoments[:, cnt] = np.std(y, axis=1)
    centralMomentsInc[:, cnt] = np.std(dy, axis=1)
    cnt = cnt + 1

    centralMoments[:, cnt] = central_moment(y, 3)
    centralMomentsInc[:, cnt] = central_moment(dy, 3)
    cnt = cnt + 1

    centralMoments[:, cnt] = central_moment(y, 4)
    centralMomentsInc[:, cnt] = central_moment(dy, 4)
    cnt = cnt + 1

print('Computation Finished: Central Moments')

cnt = 0
for i in range(0, 6, 1):

    y = np.loadtxt(DATA_DIR + 'y_response_' + str(i) + '.txt')
    dy = np.diff(y, axis=1)

    minMax[:, cnt] = np.min(y, axis=1)
    minMaxInc[:, cnt] = np.min(y, axis=1)
    cnt = cnt + 1

    minMax[:, cnt] = np.max(y, axis=1)
    minMaxInc[:, cnt] = np.max(y, axis=1)
    cnt = cnt + 1

print('Computation Finished: Min-Max')

# ============================================
# Load y3 response data
# ============================================

y3 = np.loadtxt(DATA_DIR + 'y_response_3' + '.txt')


# ============================================
# Compute statistics from y3 response series
# ============================================
# Since y_3, y_4, y_5 are all proportional to each other, I only need to analyze y3

# Initialize arrays
zeroCross = np.zeros(nSample)
acor = np.zeros((nSample, 2))
SpecMom = np.zeros((nSample, 5))
pFD = np.zeros((nSample, 2))
hFD = np.zeros((nSample, 2))
kFD = np.zeros((nSample, 2))
DFA = np.zeros((nSample, 2))
pEntropy = np.zeros((nSample, 2))
aEntropy = np.zeros((nSample, 2))
sEntropy = np.zeros((nSample, 2))
sEntropy_2 = np.zeros((nSample, 2))
svdEntropy = np.zeros((nSample, 2))
spEntropy = np.zeros((nSample, 2))
Hurst = np.zeros((nSample, 2))

for n in range(nSample):
    y_n = y3[n, :]
    y_n = y_n - np.mean(y_n)
    dy_n = np.diff(y_n)

    # Zero Crossings
    zeroCross[n] = ant.num_zerocross(y_n)

    # Autocorrelations
    acor[n, :] = lag_autocor(y_n)

    # Spectral Moments
    yd = scipy.signal.detrend(y_n)
    SpecMom[n, :] = compute_spectral_moments(yd)

    # Petrosian fractal dimension
    pFD[n, 0] = ant.petrosian_fd(y_n)
    pFD[n, 1] = ant.petrosian_fd(dy_n)

    # Higuchi fractal dimension
    hFD[n, 0] = ant.higuchi_fd(y_n)
    hFD[n, 1] = ant.higuchi_fd(dy_n)

    # Higuchi fractal dimension
    kFD[n, 0] = ant.katz_fd(y_n)
    kFD[n, 1] = ant.katz_fd(dy_n)

    # Detrended fluctuation analysis
    DFA[n, 0] = ant.detrended_fluctuation(y_n)
    DFA[n, 1] = ant.detrended_fluctuation(dy_n)

    # Permutation Entropy -- Seems to have close to 1 values
    pEntropy[n, 0] = ant.perm_entropy(y_n, normalize=True)
    pEntropy[n, 1] = ant.perm_entropy(dy_n, normalize=True)

    # Approximate Entropy -- Seems to have varying values
    aEntropy[n, 0] = ant.app_entropy(y_n)
    aEntropy[n, 1] = ant.app_entropy(dy_n)

    # Sample Entropy -- Seems to have varying values + inf
    clip_max = 5
    sEntropy[n, 0] = np.clip(ant.sample_entropy(y_n), 0, clip_max)
    sEntropy[n, 1] = np.clip(ant.sample_entropy(dy_n), 0, clip_max)

    # SVD Entropy -- Seems to have close to 1 values
    svdEntropy[n, 0] = ant.svd_entropy(y_n, normalize=True)
    svdEntropy[n, 1] = ant.svd_entropy(dy_n, normalize=True)

    # Spectral Entropy
    spEntropy[n, 0] = ant.spectral_entropy(y_n, sf=1, method='welch', nperseg=60, normalize=True)
    spEntropy[n, 1] = ant.spectral_entropy(dy_n, sf=1, method='welch', nperseg=59, normalize=True)

    # Rescale-range Analysis
    Hurst[n, 0] = nolds.hurst_rs(y_n)
    Hurst[n, 1] = nolds.hurst_rs(dy_n)

    if np.mod(n, 100) == 0:
        print('Percentage of Computation finished:', n/nSample * 100)


# ============================================
# Load y3, y4, & y5 response data
# ============================================

y3 = np.loadtxt(DATA_DIR + 'y_response_3.txt')
y4 = np.loadtxt(DATA_DIR + 'y_response_4.txt')
y5 = np.loadtxt(DATA_DIR + 'y_response_5.txt')


# ============================================
# Compute statistics from y3, y4, & y5 series
# ============================================

# Initialize arrays
ratio = np.zeros((nSample, 6))

for i in range(nSample):

    ratio[i, 0], ratio[i, 1] = compute_slope_intercept_scipy(y3[i, :], y4[i, :])
    ratio[i, 2], ratio[i, 3] = compute_slope_intercept_scipy(y3[i, :], y5[i, :])
    ratio[i, 4], ratio[i, 5] = compute_slope_intercept_scipy(y4[i, :], y5[i, :])

print('Computation Finished: Ratio')


# ============================================
# Combine all features
# ============================================
zeroCross = zeroCross.reshape(nSample, 1)
X_Base = np.hstack((centralMoments, centralMomentsInc, minMax, minMaxInc,
                       zeroCross, acor, SpecMom, pFD, hFD, kFD, DFA,
                       pEntropy, aEntropy, sEntropy, svdEntropy, spEntropy,
                       Hurst, ratio))


# ============================================
# Save statistics
# ============================================

if optTrnTst == 1:

    X = np.loadtxt(DATA_DIR + 'input.txt', delimiter=',')
    YTrnVal = X[:, 0:5]

    # Add both the aleatoric and the first 2 epistemic variables as input
    X_Step4 = np.hstack((X_Base, X[:, 0:4]))

    np.save(ExtractedDATA_DIR + 'XTrnVal_centralMoments', centralMoments)
    np.save(ExtractedDATA_DIR + 'XTrnVal_centralMomentsInc', centralMomentsInc)
    np.save(ExtractedDATA_DIR + 'XTrnVal_minMax', minMax)
    np.save(ExtractedDATA_DIR + 'XTrnVal_minMaxInc', minMaxInc)
    np.save(ExtractedDATA_DIR + 'XTrnVal_zeroCross', zeroCross)
    np.save(ExtractedDATA_DIR + 'XTrnVal_acor', acor)
    np.save(ExtractedDATA_DIR + 'XTrnVal_SpecMom', SpecMom)
    np.save(ExtractedDATA_DIR + 'XTrnVal_pFD', pFD)
    np.save(ExtractedDATA_DIR + 'XTrnVal_hFD', hFD)
    np.save(ExtractedDATA_DIR + 'XTrnVal_kFD', kFD)
    np.save(ExtractedDATA_DIR + 'XTrnVal_DFA', DFA)
    np.save(ExtractedDATA_DIR + 'XTrnVal_pEntropy', pEntropy)
    np.save(ExtractedDATA_DIR + 'XTrnVal_aEntropy', aEntropy)
    np.save(ExtractedDATA_DIR + 'XTrnVal_sEntropy', sEntropy)
    np.save(ExtractedDATA_DIR + 'XTrnVal_svdEntropy', svdEntropy)
    np.save(ExtractedDATA_DIR + 'XTrnVal_spEntropy', spEntropy)
    np.save(ExtractedDATA_DIR + 'XTrnVal_Hurst', Hurst)
    np.save(ExtractedDATA_DIR + 'XTrnVal_ratio', ratio)

    np.save(ExtractedDATA_DIR + 'XTrnVal_Base', X_Base)
    np.save(ExtractedDATA_DIR + 'XTrnVal_Step4', X_Step4)
    np.save(ExtractedDATA_DIR + 'YTrnVal', YTrnVal)

else:

    np.save(ExtractedDATA_DIR + 'XDNV_centralMoments', centralMoments)
    np.save(ExtractedDATA_DIR + 'XDNV_centralMomentsInc', centralMomentsInc)
    np.save(ExtractedDATA_DIR + 'XDNV_minMax', minMax)
    np.save(ExtractedDATA_DIR + 'XDNV_minMaxInc', minMaxInc)
    np.save(ExtractedDATA_DIR + 'XDNV_zeroCross', zeroCross)
    np.save(ExtractedDATA_DIR + 'XDNV_acor', acor)
    np.save(ExtractedDATA_DIR + 'XDNV_SpecMom', SpecMom)
    np.save(ExtractedDATA_DIR + 'XDNV_pFD', pFD)
    np.save(ExtractedDATA_DIR + 'XDNV_hFD', hFD)
    np.save(ExtractedDATA_DIR + 'XDNV_kFD', kFD)
    np.save(ExtractedDATA_DIR + 'XDNV_DFA', DFA)
    np.save(ExtractedDATA_DIR + 'XDNV_pEntropy', pEntropy)
    np.save(ExtractedDATA_DIR + 'XDNV_aEntropy', aEntropy)
    np.save(ExtractedDATA_DIR + 'XDNV_sEntropy', sEntropy)
    np.save(ExtractedDATA_DIR + 'XDNV_svdEntropy', svdEntropy)
    np.save(ExtractedDATA_DIR + 'XDNV_spEntropy', spEntropy)
    np.save(ExtractedDATA_DIR + 'XDNV_Hurst', Hurst)
    np.save(ExtractedDATA_DIR + 'XDNV_ratio', ratio)

    np.save(ExtractedDATA_DIR+'XDNV_Base', X_Base)
