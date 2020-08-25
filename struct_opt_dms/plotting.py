'''
This file is a modification of functions from DM_phase.py by Daniele Michilli
(https://github.com/danielemichilli/DM_phase/).

I have formatted and modified a number of parts to enable work with Python 3
and provide interaction in a jupyter notebook.

- D. Vohl, August 2020.
'''

import numpy as np
from .dm_phase import fit_power
import builtins

def get_yticks(freq_id_low, freq_id_high):
    return np.linspace(freq_id_low  - 0.5,
                       freq_id_high + 0.5,
                       9)

def get_yticklabels(f_channels, freq_id_low, freq_id_high):
    df = np.median(np.diff(f_channels))

    return np.round(
        np.linspace(
            f_channels[freq_id_low - freq_id_low] - df / 2.,
            f_channels[(freq_id_high - freq_id_low) - 1] + df / 2.,
            9
        ),
        1
    )

def get_xticks(t0, t1):
    return np.linspace(t0,
                       t1,
                       5)

def get_xticklabels(t0, t1, dt):
    return ['%.1f'  % (i) for i in np.linspace(
            (-(t0 - (t1/2)) * dt) * 1000,
            (-((t1/2) - t0) * dt) * 1000,
            5
        )]

def plot_coherent_power(power_vs_dm,
                        d_power_vs_dm,
                        dm_trials,
                        f_channels,
                        nchan,
                        estimated_dm,
                        dt,
                        delta_dm,
                        t0,
                        t1,
                        fluct_id_low,
                        fluct_id_high,
                        ax_power,
                        ax_power_prof,
                        ax_power_res,
                        cmap='viridis'):
    """Plot coherent power: fluctuation freq. vs DM"""

    returns_poly, dm_trials, dm_curve, _range, snr, x, y = fit_power(dm_trials,
                                                                     f_channels,
                                                                     nchan,
                                                                     d_power_vs_dm,
                                                                     fluct_id_low,
                                                                     fluct_id_high)

    # Profile
    X, Y = dm_trials, dm_curve
    ax_power_prof.plot(X, Y, linewidth=3, clip_on=False)
    ax_power_prof.plot(X[_range],
                       np.polyval(returns_poly[2], X[_range]),
                       color='orange',
                       linewidth=3,
                       zorder=2,
                       clip_on=False)
    ax_power_prof.set_xlim([X.min(), X.max()])
    ax_power_prof.set_ylim([Y.min(), Y.max()])
    ax_power_prof.ticklabel_format(useOffset=False)

    ax_power_prof.text(0.1, 0.8,
                       'S/N=%.2f' % (snr),
                       horizontalalignment='center',
                       verticalalignment='center',
                       transform=ax_power_prof.transAxes)

    # Residuals
    res = y - np.polyval(returns_poly[2], x)
    res -= res.min()
    res /= res.max()

    ax_power_res.plot(x, res, 'x', linewidth=2, clip_on=False)
    ax_power_res.set_ylim([np.min(res) - np.std(res) / 2,
                           np.max(res) + np.std(res) / 2])
    ax_power_res.set_ylabel('$\Delta$')
    ax_power_res.tick_params(axis='both',
                             labelbottom='off',
                             labelleft='off',
                             direction='in',
                             left='off',
                             top='on')
    ax_power_res.ticklabel_format(useOffset=False)

    # Power vs DM map
    FT_len = power_vs_dm.shape[0]
    indx2Ang = 1. / (2 * FT_len * dt * 1000)
    extent = [np.min(X)+estimated_dm,
              np.max(X)+estimated_dm,
              fluct_id_low * indx2Ang,
              fluct_id_high * indx2Ang]

    ax_power.imshow(power_vs_dm[fluct_id_low : fluct_id_high],
                    origin='lower',
                    aspect='auto',
                    cmap=cmap,
                    extent=extent,
                    interpolation='nearest')

    builtins.power_vs_dm = power_vs_dm[fluct_id_low : fluct_id_high]

    ax_power.tick_params(axis='both',
                         direction='in',
                         right='on',
                         top='on')

    dm = returns_poly[0]
    dm_std = returns_poly[1]

    return dm, dm_std, snr

def plot_waterfall(waterfall,
                   f_channels,
                   t0,
                   t1,
                   freq_id_low,
                   freq_id_high,
                   ax_waterfall,
                   ax_t_snr,
                   ax_power_prof,
                   ax_power_res,
                   dm,
                   dt,
                   delta_dm,
                   dm_std,
                   cmap='viridis'
                  ):
    plot_wat_map = ax_waterfall.imshow(
        waterfall,
        origin='lower',
        aspect='auto',
        cmap=cmap,
        interpolation='nearest',
        extent=(t0 - 0.5,
                t1 + 0.5,
                freq_id_low  - 0.5,
                freq_id_high + 0.5)
    )

    # set time as label instead of channel numbers
    ax_waterfall.set_xticks(
        get_xticks(t0, t1)
    )
    ax_waterfall.set_xticklabels(
        get_xticklabels(t0, t1, dt),
#         rotation=90
    )

    # set frequencies as label instead of channel numbers
    ax_waterfall.set_yticks(
        get_yticks(freq_id_low, freq_id_high)
    )
    ax_waterfall.set_yticklabels(
        get_yticklabels(f_channels, freq_id_low, freq_id_high),
    )

    plot_wat_map.autoscale()

    # plot summed profile
    wat_prof = np.nansum(waterfall, axis=0)
    plot_wat_prof, = ax_t_snr.plot(wat_prof, '-', linewidth=2)
    ax_t_snr.set_ylim([wat_prof.min()-1, wat_prof.max()+1])
    ax_t_snr.set_xlim([0, wat_prof.size])
    ax_t_snr.text(0.1, 0.8,
                  r'DM=%.2f $\pm$ %.2f pc/cm$^3$' % (dm + delta_dm, dm_std),
                  horizontalalignment='center',
                  verticalalignment='center',
                  transform=ax_t_snr.transAxes)

    ax_power_prof.axis('off')
    ax_power_res.axis('off')
    ax_t_snr.axis('off')
