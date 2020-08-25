'''
Most of the code in this file originates from DM_phase.py by Daniele Michilli
(https://github.com/danielemichilli/DM_phase/).

I have formatted and modified a number of parts to enable work with Python 3
and provide interaction in a jupyter notebook.

- D. Vohl, August 2020.
'''
import numpy as np
from scipy.fftpack import fft, ifft

def dedisperse_waterfall(wfall, DM, freq, dt, ref_freq="top"):
    """Dedisperse a waterfall matrix to a given DM."""

    k_DM = 1. / 2.41e-4
    dedisp = np.zeros_like(wfall)

    # pick reference frequency for dedispersion
    if ref_freq == "top":
        reference_frequency = freq[-1]
    elif ref_freq == "center":
        center_idx = len(freq) // 2
        reference_frequency = freq[center_idx]
    elif ref_freq == "bottom":
        reference_frequency = freq[0]
    else:
        print("`ref_freq` not recognized, using 'top'")
        reference_frequency = freq[-1]

    shift = (k_DM * DM * (reference_frequency**-2 - freq**-2) / dt).round().astype(int)
    for i,ts in enumerate(wfall):
        dedisp[i] = np.roll(ts, shift[i])
    return dedisp

def get_cohenrent_spectrum(waterfall):
    """Get the coherent spectrum of the waterfall."""

    ft_waterfall = fft(waterfall)
    amp = np.abs(ft_waterfall)
    amp[amp == 0] = 1
    spect = np.sum(ft_waterfall / amp, axis=0)
    return spect

def get_coherent_power(waterfall):
    """Get the coherent power of the waterfall."""

    spectra = get_cohenrent_spectrum(waterfall)
    power = np.abs(spectra)**2
    return power

def poly_max(x, y, Err):
    """
    Polynomial fit
    """
    n = np.linalg.matrix_rank(np.vander(y))
    p = np.polyfit(x, y, n)
    Fac = np.std(y) / Err

    dp      = np.polyder(p)
    ddp     = np.polyder(dp)
    cands   = np.roots(dp)
    r_cands = np.polyval(ddp, cands)
    first_cut = cands[(cands.imag==0) &
                      (cands.real>=min(x)) &
                      (cands.real<=max(x)) &
                      (r_cands<0)]

    if first_cut.size > 0:
        Value     = np.polyval(p, first_cut)
        Best      = first_cut[Value.argmax()]
        delta_x   = np.sqrt(np.abs(2 * Err / np.polyval(ddp, Best)))
    else:
        Best    = 0.
        delta_x = 0.

    return float(np.real(Best)), delta_x, p , Fac

def fit_power(dm_trials, f_channels, nchan, d_power_vs_dm, fluct_id_low, fluct_id_high):
    dm_curve = d_power_vs_dm[fluct_id_low : fluct_id_high].sum(axis=0)

    fact_idx = fluct_id_low - fluct_id_high
    _max   = dm_curve.max()
    _nchan = len(f_channels)
    _mean  = nchan              # Base on Gamma(2,)
    _std   = _mean / np.sqrt(2)  # Base on Gamma(2,)
    m_fact = np.sum(np.arange(fluct_id_low, fluct_id_high)**2)
    s_fact = np.sum(np.arange(fluct_id_low, fluct_id_high)**4)**0.5
    d_mean = _mean * m_fact
    d_std  = _std  * s_fact
    snr    = (_max - d_mean) / d_std

    _peak  = dm_curve.argmax()
    _range = np.arange(_peak - 5, _peak + 5)
    y = dm_curve[_range]
    x = dm_trials[_range]
    returns_poly = poly_max(x, y, d_std)

    return returns_poly, dm_trials, dm_curve, _range, snr, x, y
