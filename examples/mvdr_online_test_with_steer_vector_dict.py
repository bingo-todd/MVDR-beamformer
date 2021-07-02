import os
import numpy as np
import matplotlib.pyplot as plt

from BasicTools import plot_tools

import sys
sys.path.append('../MVDR')
from mvdr_online import MVDR  # noqa:E402


eg_dir = 'online_with_steer_vector_dict'
os.makedirs(eg_dir, exist_ok=True)


def test1(azi, alpha):
    mic_dist = [0, 0.18]
    frame_len = 320
    frame_shift = 160
    fs = 16000
    mvdr = MVDR(
        mic_dist, frame_len, frame_shift, fs, alpha=alpha,
        steer_vector_dict_path='steer_vector_dict-Anechoic.pkl')
    mvdr.plot_spatial_rp(
        azi=azi, f=1000, init_R_bins=True,
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

    src1_azi = -20
    src2_azi = 20

    # decay_coef = 0.9512  # time constant: 20 frame
    decay_coef = 0.8187  # time constant: 5 frame

    suffix = f'alpha_{alpha}-decay_ceof_{decay_coef:.4f}'

    record_1, fs = wav_tools.read('Anechoic-snr_0-14_0.wav')
    record_2, fs = wav_tools.read('Anechoic-snr_0-14_0_22.wav')

    mix_len = np.min([record_1.shape[0], record_2.shape[0]])
    mix = record_1[:mix_len, :]+record_2[:mix_len, :]
    wav_tools.write(mix, fs, f'{eg_dir}mix.wav')
    # amp_max = np.max(np.abs(mix))

    mix = np.pad(mix, [[frame_len, frame_len], [0, 0]])
    mix_len = mix.shape[0]

    mic_dist = [0, 0.18]
    fs = 16000
    mvdr_src1 = MVDR(
        mic_dist, frame_len, frame_shift, fs, decay_coef=decay_coef,
        alpha=alpha,
        steer_vector_dict_path='steer_vector_dict-Anechoic.pkl')
    mvdr_src2 = MVDR(
        mic_dist, frame_len, frame_shift, fs, decay_coef=decay_coef,
        alpha=alpha,
        steer_vector_dict_path='steer_vector_dict-Anechoic.pkl')

    n_frame = np.int(np.floor((mix_len-frame_len)/frame_shift)+1)
    rp_src1_all = np.zeros((n_frame, 37))
    rp_src2_all = np.zeros((n_frame, 37))
    for frame_i in range(n_frame):
        frame_start = frame_i*frame_shift
        frame_end = frame_start + frame_len
        frame_slice = slice(frame_start, frame_end)
        mvdr_src1.run(mix[frame_slice], src1_azi)
        mvdr_src2.run(mix[frame_slice], src2_azi)

        if np.mod(frame_i, 1) == 0:
            rp_src1_all[frame_i] = \
                mvdr_src1.plot_spatial_rp(src1_azi, init_R_bins=True)
            rp_src2_all[frame_i] = \
                mvdr_src2.plot_spatial_rp(src2_azi, init_R_bins=True)

            # fig, ax = plot_tools.subplots(3, 2)
            # ax[0, 0].plot(record_1[frame_len:][frame_slice])
            # ax[0, 0].set_ylim([-amp_max, amp_max])
            # ax[0, 1].plot(record_2[frame_len:][frame_slice])
            # ax[0, 1].set_ylim([-amp_max, amp_max])
            # ax[1, 0].plot(mix[frame_slice])
            # ax[1, 0].set_ylim([-amp_max, amp_max])
            # ax[2, 0].plot(rp_src1_all[frame_i])
            # ax[2, 1].plot(rp_src2_all[frame_i])
            # fig.savefig(f'{eg_dir}/frame_{frame_i:0>4}.png')
            # plt.close(fig)

    src1_enhanced = mvdr_src1.get_output()
    src1_enhanced = src1_enhanced[frame_len:-frame_len]
    src2_enhanced = mvdr_src2.get_output()
    src2_enhanced = src2_enhanced[frame_len:-frame_len]

    fig, ax = plot_tools.subplots(4, 2)
    ax[0, 0].plot(record_1)
    ax[0, 0].set_ylabel('src1')
    ax[0, 1].plot(record_2)
    ax[0, 1].set_ylabel('src2')
    ax[0, 0].set_ylabel('record')

    ax[1, 0].plot(mix)
    ax[1, 0].set_title('mix')

    ax[2, 0].imshow(rp_src1_all.T, aspect='auto', vmin=0, vmax=3)
    ax[2, 1].imshow(rp_src2_all.T, aspect='auto', vmin=0, vmax=3)
    ax[2, 0].set_ylabel('rp')

    ax[3, 0].plot(src1_enhanced)
    ax[3, 1].plot(src2_enhanced)
    ax[3, 0].set_ylabel('record_extracted')

    wav_tools.write(
        src1_enhanced, fs,
        f'{eg_dir}/src1_enhanced-{suffix}-{alpha:.2e}.wav')
    wav_tools.write(
        src2_enhanced, fs,
        f'{eg_dir}/src2_enhanced-{suffix}-{alpha:.2e}.wav')

    fig.savefig(f'{eg_dir}/eg-{suffix}.png')
    plt.close()


if __name__ == '__main__':
    for azi in [0, 30, 60]:
        test1(azi, alpha=1e-6)

    test2(alpha=1e-6)
