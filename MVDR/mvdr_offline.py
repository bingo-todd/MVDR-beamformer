import numpy as np
import pickle
import matplotlib.pyplot as plt


from BasicTools import fft
from BasicTools import plot_tools


class MVDR:
    """
    """
    def __init__(self, mic_dist, frame_len, frame_shift, fs, c=343,
                 delay_dict_path=None, alpha=1e-6):
        """
        Args:
            mic_dist: distances between adjcent mics, in shape of [n_mic],
                start with 0
            frame_len, frame_shift: params of fft
            fs: sample frequency
            delay_dict_path: pkl file path, which contain {azi: delays}
            c: sound speed, default to 343m/s
            alpha: weight of eye matrix added to correlation matrix
        """
        self.mic_dist = mic_dist
        self.n_mic = len(mic_dist)

        self.fs = fs
        self.frame_len = frame_len
        self.frame_shift = frame_shift

        self.n_freq_bin = frame_len
        self.n_freq_bin_valid = np.int(np.floor(self.n_freq_bin/2.0)+1)
        self.freq_bins = np.arange(self.n_freq_bin)/self.n_freq_bin*self.fs
        # valid frequency range
        self.freq_bins_valid = self.freq_bins[:self.n_freq_bin_valid]
        self.angular_freq_valid = 2*np.pi*self.freq_bins_valid

        delay_dict = None
        if delay_dict_path is not None:
            with open(delay_dict_path, 'rb') as f:
                delay_dict = pickle.load(f)
        self.delay_dict = delay_dict

        self.c = c  # sound seed
        self.alpha = alpha
        self.EPS = 1e-20

    def cal_delays(self, azi):
        """ given DOA, calculate delay of each mic(use mic0 as ref):
                >0 = ahead
                <0 = behind
        """
        if self.delay_dict is not None:
            delays = self.delay_dict[f'{int(azi)}']
        else:
            mic_dist_cumsum = np.cumsum(self.mic_dist)
            azi_rad = np.deg2rad(azi-90)
            delays = (mic_dist_cumsum*np.cos(azi_rad)/self.c)
        return delays

    def cal_steer_vector(self, azi):
        """
        """
        # time delays
        delays = self.cal_delays(azi)
        # phase delays, in shape of [n_freq_bin, n_mic]
        phase_delays =\
            np.exp(1j*self.angular_freq_valid[:, np.newaxis]
                   * delays[np.newaxis, :])
        # amplitude normalization
        steer_vector = \
            (phase_delays
             / np.sqrt(
                 np.sum(
                     phase_delays*np.conj(phase_delays), axis=1,
                     keepdims=True)))
        return steer_vector

    def cal_correlation_matrix(self, x_stft):
        """
        """
        n_frame, n_freq_bin, n_chann = x_stft.shape
        R_bins = np.zeros((n_freq_bin, n_chann, n_chann), dtype=x_stft.dtype)
        for freq_bin_i in range(n_freq_bin):
            R_bins[freq_bin_i] =\
                np.matmul(
                    x_stft[:, freq_bin_i, :].T,
                    np.conj(x_stft[:, freq_bin_i, :]))
            R_bins[freq_bin_i] = R_bins[freq_bin_i]+self.alpha*np.eye(n_chann)
            # it is vital to add the eye matrix
        return R_bins/n_frame

    def cal_h(self, R_bins, azi):
        steer_vector = self.cal_steer_vector(azi)
        n_freq_bin, n_chann, _ = R_bins.shape
        h_bins = np.zeros((n_freq_bin, n_chann), dtype=R_bins.dtype)
        # DC part
        h_bins[0, :] = 1.0/n_chann
        for i in range(1, n_freq_bin):
            R_inverse = np.linalg.pinv(R_bins[i])
            h_bins[i] = \
                (np.matmul(R_inverse, steer_vector[i, :, np.newaxis])[:, 0]
                 / (np.matmul(
                     np.matmul(
                         np.conj(steer_vector[i, np.newaxis, :]), R_inverse),
                     steer_vector[i, :, np.newaxis])+self.EPS))
        if np.mod(self.frame_len, 2) == 0:
            # fs/2
            h_bins[-1, :] = 1/n_chann
        return h_bins

    def filter(self, x, azi, plot_spatial_rp=False, spatial_rp_path=None):
        """
        """
        x = np.pad(x, [[self.frame_len, self.frame_len], [0, 0]])
        x_stft = fft.cal_stft(
            x, frame_len=self.frame_len, frame_shift=self.frame_shift)
        n_frame, n_freq_bin, n_chann = x_stft.shape

        R_bins = self.cal_correlation_matrix(x_stft)
        if plot_spatial_rp:
            self.plot_spatial_rp(
                azi, R_bins=R_bins, fig_path=spatial_rp_path)

        h_bins = self.cal_h(R_bins, azi)

        x_stft_filtered = \
            np.zeros([n_frame, n_freq_bin, 1], dtype=x_stft.dtype)
        for freq_bin_i in range(n_freq_bin):
            x_stft_filtered[:, freq_bin_i, 0] = \
                np.matmul(
                    x_stft[:, freq_bin_i, :], np.conj(h_bins[freq_bin_i]))
        x_mvdr = fft.cal_istft(
            x_stft_filtered,
            frame_len=self.frame_len, frame_shift=self.frame_shift,
            norm_win=True)
        return x_mvdr[:, 0][self.frame_len:-self.frame_len]

    def plot_spatial_rp(self, azi, init_R_bins=False, f=1e3,
                        azi_left=-90, azi_right=90, azi_step=5,
                        fig_path=None):
        """
        """
        azi_all = np.arange(azi_left, azi_right+azi_step, azi_step)
        n_azi = azi_all.shape[0]

        if not init_R_bins:
            R_bins = self.R_bins
        else:
            steer_vector = self.cal_steer_vector(azi)
            R_bins = \
                self.cal_correlation_matrix(steer_vector[np.newaxis, :, :])

        h_bins = self.cal_h(R_bins, azi)

        freq_bin_i = np.argmin(np.abs(self.freq_bins_valid-f))
        f = self.freq_bins_valid[freq_bin_i]

        rp = np.zeros(n_azi)  # response power
        for i in range(n_azi):
            tmp_steer_vector = self.cal_steer_vector(azi_all[i])
            tmp_R_bins = \
                self.cal_correlation_matrix(tmp_steer_vector[np.newaxis, :, :])
            # spatial scan
            rp[i] = \
                np.real(
                    np.matmul(
                        np.matmul(
                            np.conj(h_bins[freq_bin_i]),
                            tmp_R_bins[freq_bin_i]),
                        h_bins[freq_bin_i]))
            rp[i] = rp[i]/np.max(np.abs(np.diag(tmp_R_bins[freq_bin_i])))
        fig, ax = plot_tools.subplots(1, 1)
        ax.plot(azi_all, rp)
        # ax.set_yscale('log')
        ax.set_title(f'f: {f} Hz')
        fig.savefig(fig_path)
        plt.close()
