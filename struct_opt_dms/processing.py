import numpy as np
from blimpy import Waterfall
from .extern.psrpy.spectra import Spectra
from .extern.time_domain_astronomy_sandbox.backend import Backend

"""
 Note:
    Backend() currently is only used with default inputs,
    which means it loads the default settings for ARTS
    (which observes at L-Band). For other backends, it will
    need to be modified by calling the constructor with your
    backend's specificities.

    See documentation of time_domain_astronomy_sandbox for more information.
    https://time-domain-astronomy-sandbox.readthedocs.io
"""

def read_filterbank(filename:str,
                    t_res:float = Backend().sampling_time,
                    f_channels:list = Backend().frequencies[::-1],
                    output_type:str='spectra'):

    data = Waterfall(filename).data[:,0,:].T[::-1, :]

    if output_type == 'spectra':
        return Spectra(f_channels,
                       t_res,
                       data)
    elif output_type  == 'observation':
        return Observation(backend=Backend(),
                           length=data.data.shape[1]*data.dt,
                           window=data.data)

def zoom_around_peak(spectra:Spectra,
                     t_zoom:float = 1.):
    peak_ind = np.argmax(spectra.data.sum(axis=0))
    n_samp = int(np.round(t_zoom / spectra.dt))
    samp_start = int(peak_ind - 0.5 * n_samp)
    return data.data[:, samp_start:samp_start + n_samp]

def get_dm_trials(estimated_dm:float = 349.2,
                  dm_step:float = 0.1,
                  dm_range:int = 5):
    return np.arange(estimated_dm - dm_range,
                     estimated_dm + dm_range + .5 * dm_step, dm_step)

def correct_bandpass(spectra:Spectra):
    """Liam Connor's correct_bandpass"""
    return spectra.data - np.median(spectra.data, axis=1, keepdims=True)

def crop(spectra:Spectra,
         t_zoom:float = 0.25,
         around_peak=True):

    n_samp = int(np.round(t_zoom / spectra.dt))
    if around_peak:
        peak_ind = np.argmax(np.median(spectra.data, axis=0))
        start = int(np.round(peak_ind - (0.5 * n_samp)))
    else:
        start = int(np.round(spectra.data.shape[1]//2 - (0.5 * n_samp)))

    if start < 0:
        n_samp += start
        start = 0

    return spectra.data[:, start:start+n_samp]

def to_snr(spectra:Spectra, axis=1):
    data = spectra.data
    data = data - np.nanmean(data, axis=axis)[:, None]
    data = data / np.sqrt(np.nanvar(data, axis=axis))[:, None]
    data[~np.isfinite(data)] = np.nanmedian(data)
    return data

def acf(x):
    l = 2 ** int(np.log2(x.shape[1] * 2 - 1))
    fftx = np.fft.fft(x, n = l, axis = 1)
    ret = np.fft.ifft(fftx * np.conjugate(fftx), axis = 1)
    ret = np.fft.fftshift(ret, axes=1)
    return ret

def subband(data, sub_factor, dim='freq'):
    nfreq, nsamp = data.shape
    return np.nansum(
        data.reshape(-1, sub_factor, nsamp) if dim == 'freq' else \
        data.reshape(nfreq, sub_factor, -1, order='f'),
        axis=1
    )
