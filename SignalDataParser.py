#!/usr/bin/python
# -*- coding: utf-8 -*-
import sys
import os
import re
import numpy as np
import lxml.etree as etree
import SignalDataInfo as sdi


class SignalDataParser(object):
    def __init__(self):
        self._xml_file = None
        # self._tree = None
        # self._root = None
        # self._NSMAP = None
        # self.export_file_name = None
        # self.sampling_frequency = None
        # self.first_sample_timestamp = None
        # self.calibration_gain = None
        # self.calibration_offset = None
        # self.labels = None

    @staticmethod
    def read_data(xml_file):
        with open(xml_file, 'rb')as file:
            path, _ = os.path.split(xml_file)
            tree = etree.parse(file)
            root = tree.getroot()
            ns = re.match('^\{([^\}]+)\}', root.tag).group(1)
            nsmap = {'rs': ns}
            export_file_name = tree.xpath("//rs:exportFileName", namespaces=nsmap)[0].text
            source_file_format = tree.xpath("//rs:sourceFileFormat", namespaces=nsmap)[0].text
            sampling_frequency = float(tree.xpath("//rs:samplingFrequency", namespaces=nsmap)[0].text)
            channel_count = int(tree.xpath("//rs:channelCount", namespaces=nsmap)[0].text)
            sample_count = int(tree.xpath("//rs:sampleCount", namespaces=nsmap)[0].text)
            sample_type = np.float32 if 'float' in tree.xpath("//rs:sampleType", namespaces=nsmap)[0].text or \
                                        'FLOAT' in tree.xpath("//rs:sampleType", namespaces=nsmap)[
                                            0].text else np.float32
            page_size = float(tree.xpath("//rs:pageSize", namespaces=nsmap)[0].text)
            blocks_per_page = float(tree.xpath("//rs:blocksPerPage", namespaces=nsmap)[0].text)
            channel_labels = [label.text for label in tree.xpath("//rs:label", namespaces=nsmap)]
            first_sample_timestamp = float(tree.xpath("//rs:firstSampleTimestamp", namespaces=nsmap)[0].text)
            calibration_gain = [float(gain.text) for gain in tree.xpath("//rs:calibrationGain/rs:calibrationParam", namespaces=nsmap)]
            calibration_offset = [float(offset.text) for offset in
                                  tree.xpath("//rs:calibrationOffset/rs:calibrationParam", namespaces=nsmap)]

            return sdi.SignalDataInfo(export_file_name, path, source_file_format, sampling_frequency,
                                                 channel_count, sample_count, sample_type, page_size, blocks_per_page,
                                                 channel_labels, calibration_gain, calibration_offset,
                                                 first_sample_timestamp)


if __name__ == '__main__':
    sys.stdout.write('Ten plik jest modułem i nie może być uruchamiany samodzielnie!')
    exit()
