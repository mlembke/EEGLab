# -*- coding: utf-8 -*-
import sys
import os

import numpy as np
import scipy.signal as ss

from Lab7 import mtmvar


class SignalDataInfo(object):
    def __init__(self, export_file_name, path, source_file_format, sampling_frequency,
                 channel_count, sample_count, sample_type, page_size, blocks_per_page, channel_labels,
                 calibration_gain, calibration_offset, first_sample_timestamp):
        self._export_file_name = export_file_name
        self._source_file_format = source_file_format
        self._sampling_frequency = sampling_frequency
        self._channel_count = channel_count
        self._sample_count = sample_count
        self._sample_type = sample_type
        self._page_size = page_size
        self._blocks_per_page = blocks_per_page
        self._channel_labels = channel_labels
        self._calibration_gain = calibration_gain
        self._calibration_offset = calibration_offset
        self._first_sample_timestamp = first_sample_timestamp
        self._signals = self.__load_signals('{0}{1}{2}'.format(path, os.sep, export_file_name), sample_type,
                                          sample_count, channel_count, calibration_gain)
        self.__labels_indices = dict(zip(channel_labels, [i for i in range(len(channel_labels))]))
        self._triggers = []
        self.__mask = np.ones(self._signals.shape[0], dtype=bool)

    def _get_signals(self):
        return self._signals

    def _set_signals(self, signals):
        self._signals = signals

    signals = property(_get_signals, _set_signals)

    @property
    def signals_masked(self):
        return self._signals[self.__mask]

    def _get_sampling_frequency(self):
        return self._sampling_frequency

    def _set_sampling_frequency(self, sampling_frequency):
        self._sampling_frequency = sampling_frequency

    sampling_frequency = property(_get_sampling_frequency, _set_sampling_frequency)

    def _get_sample_type(self):
        return self._sample_type

    def _set_sample_type(self, sample_type):
        self._sample_type = sample_type

    sample_type = property(_get_sample_type, _set_sample_type)

    def _get_channel_labels(self):
        return self._channel_labels

    def _set_channel_labels(self, channel_labels):
        self._channel_labels = channel_labels

    channel_labels = property(_get_channel_labels, _set_channel_labels)

    def __getitem__(self, index):
        if type(index) is int:
            return self._signals[index]
        elif type(index) is str:
            return self._signals[self.__labels_indices[index]]

    @property
    def channel_count(self):
        return self._channel_count

    def __str__(self):
        return str(self._signals)

    def __load_signals(self, export_file_name, sample_type, sample_count, channel_count, calibration_gain):
        signal_samples = np.fromfile(export_file_name, dtype=sample_type)
        signal_samples_array = np.reshape(signal_samples, (sample_count, channel_count)).T
        # signals = np.empty(signal_samples_array.shape, dtype=Signal)
        signals = np.empty(signal_samples_array.shape, dtype=sample_type)
        for i in range(signal_samples_array.shape[0]):
            signals[i] = signal_samples_array[i] * calibration_gain[i]
        # for i in range(signal_samples_array.shape[0]):
        #     signals[i] = Signal(signal_samples_array[i] * calibration_gain[i])
        return signals

    def set_mask(self, mask):
        if all(isinstance(x, int) for x in mask):
            self.__mask[mask] = False
        elif all(isinstance(x, str) for x in mask):
            for i in mask:
                self.__mask[self.__labels_indices[i]] = False

    def set_references(self, reference_signals):
        if type(reference_signals) == str:
            reference_signals = reference_signals.split()
        self.set_mask(reference_signals)
        i = 0
        reference = 0
        for ref in reference_signals:
            reference += self._signals[self.__labels_indices[ref]]
            i += 1
        if i > 0:
            reference /= i
        self._signals[self.__mask] -= reference

    def filter_signals(self, b, a):
        self._signals[self.__mask] = ss.filtfilt(b, a, self._signals[self.__mask])

    def szt(self, p, frequencies):
        A, V = mtmvar.mult_AR(self._signals, 3, 1)
        A[0], A[1:] = 1, -A[1:]
        z_transform = np.zeros((len(frequencies), self._signals[self.__mask].shape[0], self._signals[self.__mask].shape[0]), dtype=complex)
        for j in range(len(frequencies)):
            z = np.array(
                [np.exp(2*np.pi*1j * (frequencies[j] / self._sampling_frequency))**(-i) for i in range(p)]
            )[:, np.newaxis, np.newaxis]
            z_transform[j] = np.sum(A * z)

        # z = [np.array([np.exp(2*np.pi*1j * (frequencies[j] / self._sampling_frequency))**(-i) for i in range(p)])[:,np.newaxis,np.newaxis] for j in range(len(frequencies))]
        # z_transform = np.sum(A[np.newaxis, :] * z, axis=1)
        return z_transform

    def estimate_psd(self, p, frequencies):
        Az = self.szt(p, frequencies)

if __name__ == '__main__':
    sys.stdout.write('Ten plik jest modułem i nie może być uruchamiany samodzielnie!')
    exit()
