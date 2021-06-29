# MVDR beamfer

MVDR beamfer in python

e.g

````python
from BasicTools import wav_tools  # BasicTools is from my another repo, provide some basic functions
from BasicTools import plot_tools

# two spatialized sound, SNR=0
record_1, fs = wav_tools.read('Anechoic-snr_0-14_0.wav')
record_2, fs = wav_tools.read('Anechoic-snr_0-14_0_22.wav')
# mix
mix_len = np.min([record_1.shape[0], record_2.shape[0]])
mix = record_1[:mix_len, :]+record_2[:mix_len, :]
wav_tools.write(mix, fs, 'mix.wav')

mic_dist = [0, 0.18]  # distance(m) bewteen two adjacent mics, 0.18m is the diameter of head
frame_len = 320  #
frame_shift = 160
fs = 16000
mvdr = MVDR(mic_dist, frame_len, frame_shift, fs)
record_1_extracted = mvdr.filter(
    mix, -20, plot_spatial_rp=True,
    spatial_rp_path=f'record_1_extracted-{alpha1:.2e}-{alpha2:.2e}-spatial-rp.png')
record_2_extracted = mvdr.filter(
    mix, 20, plot_spatial_rp=True,
    spatial_rp_path=f'record_2_extracted-{alpha1:.2e}-{alpha2:.2e}-spatial-rp.png')

````

result

![](examples/offline/eg.png)

### Beam patten(1kHz)

|                          0                           |                          30                           | 60                                                    |
| :--------------------------------------------------: | :---------------------------------------------------: | ----------------------------------------------------- |
| ![](examples/offline/spatial_response-0-1.00e+06-1.00e-06.png) | ![](examples/offline/spatial_response-30-1.00e+06-1.00e-06.png) | ![](examples/offline/spatial_response-60-1.00e+06-1.00e-06.png) |

### Equations

MVDR beamer coefficients are calculated as follow[1],

![image-20210629094237911](images/mvdr_equation1.png)

[1] Benesty, Jacob, Jingdong Chen, and Yiteng Huang. 2008. Microphone Array Signal Processing. Springer Topics in Signal Processing. Berlin Heidelberg: Springer-Verlag. https://doi.org/10.1007/978-3-540-78612-2.
