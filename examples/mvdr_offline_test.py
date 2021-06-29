import os
import numpy as np
import matplotlib.pyplot as plt

from BasicTools import plot_tools

import sys
sys.path.append('../MVDR')
from mvdr_offline import MVDR  # noqa:E402


eg_dir = 'offline'
os.makedirs(eg_dir, exist_ok=True)


def test(azi, alpha):
    mic_dist = [0, 0.18]
    frame_len = 320
    frame_shift = 160
    fs = 16000
    mvdr = MVDR(mic_dist, frame_len, frame_shift, fs, alpha=alpha)
    mvdr.plot_spatial_rp(
        azi=azi,
        fig_path=f'{eg_dir}/spatial_response-{azi}-{alpha:.2e}.png')

    delay_record = []
    for azi in range(-90, 91, 1):
        delays = mvdr.cal_delays(azi)
        delay_record.append(delays[1])

    fig, ax = plot_tools.subplots(1, 1)
    ax.plot(np.arange(-90, 91, 1), delay_record)
    fig.savefig(f'{eg_dir}/delay.png')
    plt.close()


def test2(alpha):
    from BasicTools import wav_tools
    from BasicTools import plot_tools

    record_1, fs = wav_tools.read('Anechoic-snr_0-14_0.wav')
    record_2, fs = wav_tools.read('Anechoic-snr_0-14_0_22.wav')

    mix_len = np.min([record_1.shape[0], record_2.shape[0]])
    mix = record_1[:mix_len, :]+record_2[:mix_len, :]

    wav_tools.write(mix, fs, f'{eg_dir}/mix.wav')

    mic_dist = [0, 0.18]
    frame_len = 320
    frame_shift = 160
    fs = 16000
    mvdr = MVDR(mic_dist, frame_len, frame_shift, fs, alpha=alpha)
    record_1_extracted = mvdr.filter(
        mix, -20, plot_spatial_rp=True,
        spatial_rp_path=(f'{eg_dir}/record_1_extracted-'
                         + f'{alpha:.2e}-spatial-rp.png'))
    record_2_extracted = mvdr.filter(
        mix, 20, plot_spatial_rp=True,
        spatial_rp_path=(f'{eg_dir}/record_2_extracted-'
                         + f'{alpha:.2e}-spatial-rp.png'))

    fig, ax = plot_tools.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].plot(record_1)
    ax[0, 0].set_ylabel('src1')
    ax[1, 0].plot(record_2)
    ax[1, 0].set_ylabel('src2')
    ax[0, 0].set_title('record')

    ax[0, 1].plot(mix)
    ax[0, 1].set_title('mix')

    ax[0, 2].plot(record_1_extracted)
    ax[1, 2].plot(record_2_extracted)
    ax[0, 2].set_title('record_extracted')

    wav_tools.write(
        record_1_extracted, fs,
        f'{eg_dir}/record_1_extracted-{alpha:.2e}.wav')
    wav_tools.write(
        record_2_extracted, fs,
        f'{eg_dir}/record_2_extracted-{alpha:.2e}.wav')

    fig.savefig(f'{eg_dir}/eg.png')
    plt.close()


if __name__ == '__main__':
    if True:
        for alpha in [1, 1e-3, 1e-6]:
            for azi in [0, 30, 60]:
                test(azi, alpha)

    if True:
        for alpha in [1, 1e-3, 1e-6]:
            test2(alpha)
