"""Observation class."""
import numpy as np
from .backend import Backend
from .pulse import Pulse
from .rfim import RFIm

# TODO: downsampling (see version in notebook)
class Observation():
    """Observation class."""

    def __init__(self,
                 backend:Backend,
                 length:int = 1, # length in second
                 t0:float = 0.,
                 window:np.array = None
                ):
        """Initialise Observation class.

        Parameters
        ----------
        backend : Backend
            An instance of Backend class
        length : int
            Length of the observation (in second)
        t0 : float
            Starting time of the observation (in second)

        """
        self.backend = backend
        self.length = length
        self.t0 = t0
        if window is None:
            self.window = np.random.normal(0, 1, (self.backend.n_channels,
                                                  int(self.length*backend.samples_per_second)))
        else:
            self.window = window[::-1, :]
        # self.window += np.abs(np.min(self.window))
        self.noise_median = np.median(self.window).copy()
        self.noise_std = np.std(self.window).copy()

        # Time
        self.time_to_index = lambda t_i : np.ceil((t_i-self.t0)/self.backend.sampling_time).astype(int)
        self.index_to_time = lambda index : index * self.backend.sampling_time + self.t0
        self.next_time = lambda i : i * self.backend.sampling_time

        self.times = np.array([self.next_time(i) for i in range(int(self.backend.samples_per_second *
                                                                    self.length))])
        self.time_indices = np.array([self.time_to_index(t) for t in self.times])
        # self.snr = lambda snr, area : (self.noise_median + snr * self.noise_std) / np.sqrt(area)
        self.snr = lambda snr, area : snr / np.sqrt(area)

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, window):
        self._window = window


    def time_cleaning(self, window=[], n_iter=1, threshold=3.25, symetric=False, keep_state=False):
        """RFI mitigation (cleaning) in time domain.

        Parameters
        ----------
        window : (list | Numpy.array)
            An observation window (to clean a specific instance of window). If empty, cleans self.window
        n_iter : int
            Number of cleaning iteration
        keep_state : bool
            Save result of cleaning to self.window

        Returns
        -------
        self.window : Numpy.array
            The cleaned window.

        """
        if len(window) == 0:
            window = self.window

        if keep_state:
            window = RFIm().tdsc_amber(window, threshold=threshold, n_iter=n_iter, symetric=symetric)
        else:
            return RFIm().tdsc_amber(window, threshold=threshold, n_iter=n_iter, symetric=symetric)

        self.window = window
        return self.window

    def frequency_cleaning(self, window=[], n_iter=1, bin_size=32, threshold=2.75, symetric=False, keep_state=False):
        """RFI mitigation (cleaning) in frequency domain.

        Parameters
        ----------
        window : (list | Numpy.array)
            An observation window (to clean a specific instance of window). If empty, cleans self.window
        n_iter : int
            Number of cleaning iteration
        keep_state : bool
            Save result of cleaning to self.window

        Returns
        -------
        self.window : Numpy.array
            The cleaned window.

        """
        if len(window) == 0:
            window = self.window

        if keep_state:
            window = RFIm().fdsc_amber(window, n_iter=n_iter, bin_size=bin_size, threshold=threshold, symetric=symetric)
        else:
            return RFIm().fdsc_amber(window, n_iter=n_iter, bin_size=bin_size, threshold=threshold, symetric=symetric)

        self.window = window
        return self.window

    def dm0_cleaning(self, window=[], threshold=3.25, keep_state=False):
        if len(window) == 0:
            window = self.window

        if keep_state:
            window = RFIm().dm0clean(window, threshold=threshold)
        else:
            return RFIm().dm0clean(window, threshold=threshold)

        self.window = window
        return self.window

    def dedisperse(self, dm, window=[]):
        """Dedisperse an observation window for a given dispersion measure (DM).

        Parameters
        ----------
        dm : int
            Dispersion measure to use for dedispersion
        window : (list | Numpy.array)
            An observation window (to clean a specific instance of window). If empty, cleans self.window

        Returns
        -------
        dedispersed_window : Numpy.array
            The dedispersed window.

        """
        if len(window) == 0:
            window = self.window

        return np.array([
            np.roll(window[i].copy(), -r) for i, r in zip(
                range(window.shape[0]),
                self.time_to_index(Pulse(self.backend).delays(dm))
            )
        ])

    def add_signal(self, signal_value, x_t0, x_t1, y_t0, y_t1):
        self.window[x_t0:x_t1, y_t0:y_t1] += signal_value

    def add_dispersed_pulse(self, dm, width, pulse_t0, snr=100, verbose=False):
        pulse = Pulse(self.backend, width=width)
        pulse_t_start = self.index_to_time(index=self.time_to_index(pulse_t0).astype(int))
        t_idx = self.time_to_index(t_i=pulse_t_start+pulse.delays(dm=dm))
        area = np.sqrt(self.backend.n_channels*pulse.width/self.backend.sampling_time)
        value = self.snr(snr, area)

        for i in range(t_idx.shape[0]):
            self.add_signal(value, i, i+1, t_idx[i], t_idx[i]+self.time_to_index(pulse.width))

        if verbose:
            print ("snr:", snr, "value: ", value, "area: ", area)

    def add_rfi(self,
                t_start=0., t_stop=0.5, t_step=0.03, t_width=0.003,
                f_start=200, f_stop=250,
                snr=50,
                verbose=False
               ):

        start = self.time_to_index(t_start)
        stop = self.time_to_index(t_stop)
        step = self.time_to_index(t_step)
        width = self.time_to_index(t_width)
        height = np.abs(f_start-f_stop)
        # area = np.sqrt((stop-start)/(step*width))
        area = np.sqrt(width*height)
        value = self.snr(snr, area)

        for t in range(start, stop, step):
            self.add_signal(value, f_start, f_stop, t, t+width)

        if verbose:
            print ("snr:", snr, "value: ", value)
