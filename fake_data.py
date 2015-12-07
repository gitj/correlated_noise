__author__ = 'gjones'

import numpy as np


class FakeKIDData(object):
    def __init__(self, num_samples=2 ** 19, num_detectors=16,
                 red_exponent=-1.0, red_amplitude=1e-15,
                 shot_amplitude_scale=1e-18, amplifier_amplitude_scale=1e-19,
                 rolloff_freq=2e3, sampling_rate=256e6 / 2 ** 14, seed=123):
        self.num_samples = num_samples
        self.num_detectors = num_detectors
        self.red_exponent = red_exponent
        self.red_amplitude = red_amplitude
        self.shot_amplitude_scale = shot_amplitude_scale
        self.amplifier_amplitude_scale = amplifier_amplitude_scale
        self.rolloff_freq = rolloff_freq
        self.sampling_rate = sampling_rate
        np.random.seed(seed)
        self.shot_amplitudes = (np.random.rand(num_detectors) + .1) * self.shot_amplitude_scale
        self.amplifier_amplitudes = (np.random.rand(num_detectors) + .1) * self.amplifier_amplitude_scale
        self.regenerate()

    def regenerate(self):
        nfft = self.num_samples * 2 + 1
        fft_freq = np.linspace(0, self.sampling_rate / 2, nfft)
        red_fft = nfft * np.sqrt(
            120 * self.red_amplitude * 2 / self.sampling_rate) * fft_freq ** self.red_exponent * np.exp(
            np.pi * 2j * np.random.rand(nfft))
        red_fft[0] = 0
        self.red_process = np.fft.irfft(red_fft)[:self.num_samples]
        shot_fft = np.random.randn(nfft, self.num_detectors) + 1j * np.random.randn(nfft, self.num_detectors)
        shot_fft = shot_fft * nfft * np.sqrt(120 * self.shot_amplitudes * 2 / self.sampling_rate)
        amplifier_fft = np.random.randn(nfft, self.num_detectors) + 1j * np.random.randn(nfft, self.num_detectors)
        amplifier_fft = amplifier_fft * nfft * np.sqrt(120 * self.amplifier_amplitudes * 2 / self.sampling_rate)
        self.total_signal = np.fft.irfft(
            (red_fft[:, None] * np.sqrt(self.shot_amplitudes / self.shot_amplitudes.max()) + shot_fft) / (
            (1 + 1j * fft_freq / (self.rolloff_freq))[:, None]) + amplifier_fft, axis=0)[:self.num_samples, :]
        self.signal_without_red_noise = np.fft.irfft(
            (shot_fft) / ((1 + 1j * fft_freq / (self.rolloff_freq))[:, None]) + amplifier_fft, axis=0)[
                                        :self.num_samples, :]