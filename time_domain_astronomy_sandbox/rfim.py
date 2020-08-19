"""RFIm class."""
import numpy as np


class RFIm():
    """RFIm class. A class for radio interference mitigation."""

    def __init__(self):
        """Initialise RFIm class."""
        pass

    def dm0clean(self, data, threshold=3.25):
        dtmean = np.mean(data, axis=1)
        dfmean = np.mean(data, axis=0)
        stdevf = np.std(dfmean)
        medf = np.median(dfmean)
        maskf = np.where(np.abs(dfmean - medf) > threshold*stdevf)[0]
        # replace with mean spectrum
        data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]
        return data

    def fdsc(self, data, bin_size=32, threshold=2.75):
        """Frequency domain sigma cut.

        (Modified code from https://github.com/liamconnor/arts-analysis/blob/master/triggers.py)

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality

        """
        dtmean = np.mean(data, axis=-1)
        dtmean_nobandpass = data.mean(1) - dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)
        stdevt = np.std(dtmean_nobandpass)
        medt = np.mean(dtmean_nobandpass)
        maskt = np.abs(dtmean_nobandpass - medt) > threshold*stdevt
        data[maskt] = np.median(dtmean)
        return data

    def fdsc_old(self, data, bin_size=32, threshold=2.75, n_iter=1):
        """Frequency domain sigma cut.

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality
        n_iter : int
            Number of cleaning iteration

        """
        for j in range(n_iter):
            dtmean = np.mean(data, axis=-1)
            for i in range(data.shape[1]):
                dtmean_nobandpass = data[:, i] - dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)
                stdevt = np.std(dtmean_nobandpass)
                # medt = np.median(dtmean_nobandpass)
                medt = np.mean(dtmean_nobandpass)
                maskt = np.abs(dtmean_nobandpass - medt) > threshold*stdevt

                # replace with mean bin values
                data[maskt, i] = dtmean.reshape(-1, bin_size).mean(-1).repeat(bin_size)[maskt]

        return data

    def fdsc_amber(self, data, bin_size=32, threshold=2.75, n_iter=1, symetric=False):
        """Frequency domain sigma cut.

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality
        n_iter : int
            Number of cleaning iteration

        """
        for k in range(n_iter):
            for t in range(data.shape[1]):
                current = data[:, t]
                dtmean_nobandpass = current - current.reshape(-1, bin_size).mean(-1).repeat(bin_size)
                stdevt = np.std(dtmean_nobandpass)
                # medt = np.median(dtmean_nobandpass)
                medt = np.mean(dtmean_nobandpass)

                if symetric:
                    maskt = np.abs(dtmean_nobandpass-medt) > threshold*stdevt
                else:
                    maskt = dtmean_nobandpass > medt + threshold*stdevt

                # replace with mean bin values
                data[maskt, t] = current.reshape(-1, bin_size).mean(-1).repeat(bin_size)[maskt]

        return data

    def tdsc(self, data, threshold=3.25, n_iter=1):
        """Time domain sigma cut.

        (Modified code from https://github.com/liamconnor/arts-analysis/blob/master/triggers.py)

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality
        n_iter : int
            Number of cleaning iteration

        """
        dtmean = np.mean(data, axis=-1)
        for i in range(n_iter):
            dfmean = np.mean(data, axis=0)
            stdevf = np.std(dfmean)
            medf = np.median(dfmean)
            maskf = np.where(np.abs(dfmean - medf) > threshold*stdevf)[0]
            # replace with mean spectrum
            data[:, maskf] = dtmean[:, None]*np.ones(len(maskf))[None]

        return data

    def tdsc_amber(self, data, threshold=3.25, n_iter=1, symetric=False):
        """Time domain sigma cut as implemented in AA-ALERT RFIm.

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality
        n_iter : int
            Number of cleaning iteration

        """
        for k in range(n_iter):
            for f in range(data.shape[0]):
                dfmean = np.mean(data[f, :])
                stdevf  = np.std(data[f, :])

                if symetric:
                    maskf = np.where(np.abs(data[f, :] - dfmean) > threshold*stdevf)[0]
                else:
                    maskf = np.where(data[f, :] > dfmean + threshold*stdevf)[0]

                data[f, maskf] = dfmean

        return data

    def tdsc_per_channel(self, data, threshold=3.25, n_iter=1):
        """Time domain sigma cut.

        (Code from https://github.com/liamconnor/arts-analysis/blob/master/triggers.py)

        Parameters
        ----------
        data : Numpy.Array
            2D Array
        bin_size : int
            Size of averaging bin Size
        threshold : float
            Threshold to use for sigma cut inequality
        n_iter : int
            Number of cleaning iteration

        """
        for ii in range(n_iter):
            dtmean = np.mean(data, axis=1, keepdims=True)
            dtsig = np.std(data, axis=1)
            for nu in range(data.shape[0]):
                d = dtmean[nu]
                sig = dtsig[nu]
                maskpc = np.where(np.abs(data[nu]-d)>threshold*sig)[0]
                data[nu][maskpc] = d

        return data
