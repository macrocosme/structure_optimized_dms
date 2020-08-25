"""SNR class."""
import numpy as np

class SNR():
    """SNR class. A class for signal-to-noise computation."""
    def __init__(self):
        pass

    def simple_snr(self, a, axis=0):
        """Compute signal-to-noise ratio

        Parameters
        ----------
        a : list or numpy array
            Array of data
        axis : int
            Axis onto which compute SNR

        Returns
        -------
        vals : array of float
            Values (SNR per bin)
        """
        a = np.asanyarray(a)
        m = a.mean(axis=axis)
        m_max = m.max()
        sd = m.std()

        vals = np.where(sd == 0, 0, (m-m.mean())/sd)
        return vals
