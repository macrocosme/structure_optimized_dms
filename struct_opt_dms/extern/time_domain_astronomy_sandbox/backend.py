"""Backend class."""
import numpy as np


class Backend():
    """Defaults are currently ARTS observing properties."""

    def __init__(self,
                 n_channels: int = 1536,
                 channel_bandwidth: float = 0.1953125,  # MHz
                 fmin: float = 1219.700927734375,  # MHz
                 sampling_time: float = 0.00008192,  # second
                 samples_per_second: int = 12500):
        """Initiale of Backend class.

        Parameters
        ----------
        n_channels:int = 1536,
        channel_bandwidth:float = 0.1953125, # MHz
        fmin:float = 1219.700927734375, # MHz
        sampling_time:float = 0.00008192, # second
        samples_per_second:int = 12500

        """
        self.n_channels = n_channels
        self.channel_bandwidth = channel_bandwidth
        self.fmin = fmin
        self.fmax = self.fmin+self.n_channels*self.channel_bandwidth # MHz
        self.sampling_time = sampling_time
        self.samples_per_second = samples_per_second

        self.freq_to_index = lambda frequency : int((frequency-self.fmin)/self.channel_bandwidth)
        self.next_freq = lambda i : self.fmin + i * self.channel_bandwidth
        self.frequencies = np.array([self.next_freq(i) for i in range(self.n_channels)])
        self.freq_indices = np.array([self.freq_to_index(f) for f in self.frequencies])

        self.frequency_range_to_n_channels = lambda range : np.ceil(range/self.channel_bandwidth).astype(int)
