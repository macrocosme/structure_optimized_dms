from .extern.time_domain_astronomy_sandbox.backend import Backend
from .extern.time_domain_astronomy_sandbox.observation import Observation
from .extern.time_domain_astronomy_sandbox.rfim import RFIm

from .plotting import plot_coherent_power, plot_waterfall
from .processing import (
    get_dm_trials,
    read_filterbank,
    correct_bandpass,
    crop,
    to_snr,
    subband,
)
from .dm_phase import get_coherent_power, dedisperse_waterfall

import numpy as np
import scipy.ndimage.filters as filters

import builtins

import copy

"""
 Note:
    Backend() currently is only used with default inputs,
    which means it loads the default settings for ARTS.
    For other backends, it will need to be modified by
    calling the constructor with your backend's specificities.

    See documentation of time_domain_astronomy_sandbox for more information.
    https://time-domain-astronomy-sandbox.readthedocs.io

Note:
    The use of global variable is currently simply done
    for eased interactivity with notebook's widgets.
    There must be a better way.
"""

def initialize_observation(spectra,
                           freq_id_low = 0,
                           freq_id_high = None,
                           t0 = 0,
                           t1 = None):

    if freq_id_high is None:
        freq_id_high = spectra.data.shape[0]

    if t1 is None:
        t1 = spectra.data.shape[1]

    waterfall = spectra.data[int(freq_id_low):int(freq_id_high), int(t0):int(t1)]
    f_channels = spectra.freqs[int(freq_id_low):int(freq_id_high), ...]

    return waterfall, f_channels, freq_id_high, t1

def set_layout(fig, gs):
    # Fluctuation vs dDM
    ax_power_prof = fig.add_subplot(gs[0:4, 0:3])
    ax_power_prof.clear()

    ax_power_res = fig.add_subplot(gs[4:5, 0:3])
    ax_power_res.clear()

    ax_power = fig.add_subplot(gs[5:, 0:3])
    ax_power.clear()
    ax_power.set_xlabel(r'DM (pc/cm$^3$)')
    ax_power.set_ylabel(r'Fluctuation frequency (ms$^{-1}$)')


    # Waterfall
    ax_t_snr = fig.add_subplot(gs[0:4, 3:])
    ax_t_snr.clear()

    ax_waterfall = fig.add_subplot(gs[5:, 3:])
    ax_waterfall.clear()
    ax_waterfall.set_xlabel('Time (ms)')
    ax_waterfall.set_ylabel('Frequency (MHz)')

#     # ACF
#     ax_acf_prof = fig.add_subplot(gs[0:4, 6:])
#     ax_acf_prof.clear()

#     ax_acf = fig.add_subplot(gs[5:, 6:])
#     ax_acf.clear()
#     ax_acf.set_xlabel('Time (ms)')
#     ax_acf.set_ylabel('Frequency (MHz)')

    return ax_t_snr, ax_waterfall, ax_power_prof, ax_power, ax_power_res

def select_frequency_range(spectra,
                           dm_trials,
                           fig,
                           gs,
                           power_vs_dm,
                           d_power_vs_dm,
                           fluct_id_low = 0,
                           fluct_id_high = 30,
                           freq_id_low = 0,
                           freq_id_high = None,
                           t0 = 0,
                           t1 = None,
                           ds_freq = 1,
                           ds_time = 1,
                           delta_dm = 0,
                           smooth = 0):
    """Select a frequency range from the waterfall 2D array."""

    builtins.fluct_id_low = fluct_id_low
    builtins.fluct_id_high = fluct_id_high
    builtins.freq_id_low = freq_id_low
    builtins.freq_id_high = freq_id_high
    builtins.t0 = t0
    builtins.t1 = t1
    builtins.ds_freq = ds_freq
    builtins.ds_time = ds_time
    builtins.delta_dm = delta_dm
    builtins.smooth = smooth

    # Prep figure layout
    ax_t_snr, ax_waterfall, ax_power_prof, ax_power, ax_power_res = set_layout(fig, gs)

    # Initialize observation data
    waterfall, f_channels, freq_id_high, t1 = initialize_observation(spectra,
                                                                     freq_id_low=freq_id_low,
                                                                     freq_id_high=freq_id_high,
                                                                     t0=t0,
                                                                     t1=t1)

    dm, dm_std, snr = plot_coherent_power(filters.gaussian_filter(power_vs_dm, smooth),
                                          filters.gaussian_filter(d_power_vs_dm, smooth),
                                          dm_trials,
                                          f_channels,
                                          waterfall.shape[0],
                                          spectra.dm,
                                          spectra.dt,
                                          delta_dm,
                                          t0,
                                          t1,
                                          fluct_id_low,
                                          fluct_id_high,
                                          ax_power,
                                          ax_power_prof,
                                          ax_power_res)

    ax_power.vlines(dm + delta_dm + spectra.dm,
                    ax_power.get_ylim()[0],
                    ax_power.get_ylim()[1],
                    alpha=0.7,
                    color='red')

    builtins.struct_opt_dm = spectra.dm + delta_dm + dm
    builtins.struct_opt_dm_err = dm_std

    waterfall = dedisperse_waterfall(waterfall,
                                     delta_dm + dm,
                                     f_channels,
                                     spectra.dt)

    builtins.sub_waterfall = subband(
        subband(
            waterfall,
            ds_freq,
            dim='freq'
        ),
        ds_time,
        dim='time'
    )

    plot_waterfall(builtins.sub_waterfall,
                   f_channels,
                   t0,
                   t1,
                   freq_id_low,
                   freq_id_high,
                   ax_waterfall,
                   ax_t_snr,
                   ax_power_prof,
                   ax_power_res,
                   spectra.dm,
                   spectra.dt,
                   delta_dm + dm,
                   dm_std)

    fig.canvas.draw()
    display(fig)

def prep_power(spectra,
               dm_trials,
               freq_id_low = 0,
               freq_id_high = None,
               t0 = 0,
               t1 = None,
               verbose=False):
    if verbose:
        print ('Computing coherent power vs DM...')
        print ()
    waterfall, f_channels, freq_id_high, t1 = initialize_observation(spectra,
                                                                     freq_id_low=freq_id_low,
                                                                     freq_id_high=freq_id_high,
                                                                     t0=t0,
                                                                     t1=t1)

    # Compute coherent power vs DM
    nbin = int(np.round(waterfall.shape[1] / 2))
    # global power_vs_dm
    power_vs_dm = np.zeros([nbin, dm_trials.size])
    for i, dm in enumerate(dm_trials):
        power_vs_dm[:, i] = get_coherent_power(
            dedisperse_waterfall(waterfall,
                                 dm,
                                 f_channels,
                                 spectra.dt)
        )[:nbin]

    v = np.arange(0, nbin)
    d_power_vs_dm = power_vs_dm * v[:, np.newaxis]**2

    return power_vs_dm, d_power_vs_dm

def prep_data(file, estimated_dm, downsampling, around_peak=True, verbose=False):
    """Prepare waterfall data for analysis and plotting
    """
    if verbose:
        print ('Preprocessing data...')
        print ()

    t_res = Backend().sampling_time
    f_channels = Backend().frequencies
    dm_trials = get_dm_trials(estimated_dm = 0,
                          dm_step = 0.1,
                          dm_range = 10)

    spectra = read_filterbank(file,
                              t_res = t_res,
                              f_channels = f_channels)

    spectra.data = RFIm().dm0clean(spectra.data)
    spectra.data = correct_bandpass(spectra.data)
    spectra.data = RFIm().tdsc_amber(spectra.data)
    # spectra.data = RFIm().fdsc_amber(spectra.data)

    spectra.dedisperse(dm=estimated_dm)
    # spectra.downsample(factor=downsampling)
    # spectra.subband(spectra.data.shape[0]//4)

    spectra.data = crop(spectra,
                        # t_zoom=0.05 if downsampling < 25 else 0.1 if downsampling > 1 else 0.015,
                        0.1,
                        # 0.5,
                        around_peak = around_peak)

    spectra.data = to_snr(spectra.data)

    return spectra, dm_trials

def initialize(input_filename, estimated_dm, downsampling, around_peak=None, verbose=False):
    if verbose:
        print ('Loading data... %s' % (input_filename))
        print ()

    try:
        plt.clf()
    except:
        pass

    if around_peak is None:
        try:
            spectra, dm_trials = prep_data(input_filename,
                                           estimated_dm,
                                           downsampling,
                                           around_peak=True,
                                           verbose=verbose)
        except IndexError:
            print ('Errror with %s' % file)
            print ()
            spectra, dm_trials = prep_data(input_filename,
                                           estimated_dm,
                                           downsampling,
                                           around_peak=False,
                                           verbose=verbose)
    else:
        spectra, dm_trials = prep_data(input_filename,
                                       estimated_dm,
                                       downsampling,
                                       around_peak=around_peak,
                                       verbose=verbose)

    return spectra, dm_trials, input_filename
