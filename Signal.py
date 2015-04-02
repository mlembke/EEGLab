#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import numpy as np


class Signal(object):
    def __init__(self, samples=np.array([], dtype=np.float32)):
        self._samples = samples
        # self._label = ''
        # self._gain = 0.0
        # self._offset = 0.0

    def __str__(self):
        return str(self._samples)

    def _get_samples(self):
        return self._samples

    def _set_samples(self, samples):
        self._samples = samples

    samples = property(_get_samples, _set_samples)
    #
    # def _get_label(self):
    #     return self._label
    #
    # def _set_label(self, label):
    #     self._label = label
    #
    # label = property(_get_label, _set_label)
    #
    # def _get_gain(self):
    #     return self._gain
    #
    # def _set_gain(self, gain):
    #     self._gain = gain
    #
    # gain = property(_get_gain, _set_gain)
    #
    # def _get_offset(self):
    #     return self._offset
    #
    # def _set_offset(self, offset):
    #     self._offset = offset
    #
    # offset = property(_get_offset, _set_offset)

    def __getitem__(self, index):
        return self._samples[index]

if __name__ == '__main__':
    sys.stdout.write('Ten plik jest modułem i nie może być uruchamiany samodzielnie!')
    exit()
