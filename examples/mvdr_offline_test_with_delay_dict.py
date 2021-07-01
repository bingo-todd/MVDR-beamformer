import os
import numpy as np
import matplotlib.pyplot as plt

from BasicTools import plot_tools

import sys
sys.path.append('../MVDR')
from mvdr_offline import MVDR  # noqa:E402


eg_dir = 'offline_with_delay_dict'
os.makedirs(eg_dir, exist_ok=True)


def test(azi, alpha):
    mic_dist = [0, 0.18]
    frame_len = 320
    frame_shift = 160
    fs = 16000
    mvdr = MVDR(
        mic_dist, frame_len, frame_shift, fs, alpha=alpha,
        delay_dict_path='delay_dict-Anechoic.pkl')

    mvdr.plot_spatial_rp(
        azi=azi,
        fig_path=f'{eg_dir}/spatial_response-{azi}-{alpha:.2e}.png')

    delay_record = []
    for azi in range(-90, 95, 5):
        delays = mvdr.cal_delays(azi)
        delay_record.append(delays[1])

    fig, ax = plot_tools.subplots(1, 1)
    ax.plot(np.arange(-90, 95, 5), delay_record)
    fig.savefig(f'{eg_dir}/delay.png')
    plt.close()


def test2(alpha):
    from BasicTools import wav_tools
    from BasicTools import plot_tools

    frame_len = 320
    frame_shift = 160

    record1, fs = wav_tools.read('Anechoic-snr_0-14_0.wav')
    record2, fs = wav_tools.read('Anechoic-snr_0-14_0_22.wav')

    mix_len = np.min([record1.shape[0], record2.shape[0]])
    mix = record1[:mix_len, :]+record2[:mix_len, :]
    wav_tools.write(mix, fs, f'{eg_dir}/mix.wav')

    mix = np.pad(mix, [[frame_len, frame_len], [0, 0]])
    mix_len = mix.shape[0]

    mic_dist = [0, 0.18]
    fs = 16000
    mvdr = MVDR(mic_dist, frame_len, frame_shift, fs, alpha=alpha,
                delay_dict_path='delay_dict-Anechoic.pkl')
    src1_enhanced = mvdr.filter(
        mix, -20, plot_spatial_rp=True,
        spatial_rp_path=(f'{eg_dir}/src1_enhanced-'
                         + f'{alpha:.2e}-spatial-rp.png'))
    src1_enhanced = src1_enhanced[frame_len:-frame_len]

    src2_enhanced = mvdr.filter(
        mix, 20, plot_spatial_rp=True,
        spatial_rp_path=(f'{eg_dir}/src2_enhanced-'
                         + f'{alpha:.2e}-spatial-rp.png'))
    src2_enhanced = src2_enhanced[frame_len:-frame_len]

    fig, ax = plot_tools.subplots(2, 3, sharex=True, sharey=True)
    ax[0, 0].plot(record1)
    ax[0, 0].set_ylabel('src1')
    ax[1, 0].plot(record2)
    ax[1, 0].set_ylabel('src2')
    ax[0, 0].set_title('record')

    ax[0, 1].plot(mix)
    ax[0, 1].set_title('mix')

    ax[0, 2].plot(src1_enhanced)
    ax[1, 2].plot(src2_enhanced)
    ax[0, 2].set_title('record_extracted')

    wav_tools.write(
        src1_enhanced, fs,
        f'{eg_dir}/src1_enhanced-{alpha:.2e}.wav')
    wav_tools.write(
        src2_enhanced, fs,
        f'{eg_dir}/src2_enhanced-{alpha:.2e}.wav')

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
