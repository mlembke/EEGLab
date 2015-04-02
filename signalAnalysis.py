# -*- coding: utf-8 -*-
import copy as cp

import numpy as np
import scipy.signal as ss

from Lab7.mtmvar import mult_AR


def covariance(x, y, t):
    """
    Estymuje funkcję kowariancji sygnałów x i y, dla określonego przesunięcia t.
    Wersja wolna algorytmu.
    """
    N = len(x)
    def Shift(x, s):
        shifted = np.zeros(N)
        xm = np.mean(x)
        for i in range(N):
            if((s + i >= 0) and (s + i  < N)):
               shifted[i] = x[s + i]
        return shifted

    covariance = np.zeros(2 * t + 1)
    xm = np.mean(x)
    ym = np.mean(y)
    
    for i in range(-t, t + 1):
        slidingY = Shift(y - ym, i)
        covariance[i + t] = np.dot(slidingY, x - xm)
    return covariance / (N - 1)


def covariance_fast(x, y):
    correlation = np.correlate(x, y, 'full')
    N = len(correlation)
    covariance = correlation/ (N - 1)
    return covariance


def correlate(x, y, t):
    """
    Estymuje funkcję korelacji sygnałów x i y, dla określonego przesunięcia t.
    Wersja wolna algorytmu.
    """
    covariances = covariance(x, y, t)
    N = len(covariances)
    return covariances / ((N - 1) * (np.std(x) * np.std(y)))


def correlate_fast(x, y, t):
    correlate = np.correlate(x, y, 'full')
    N = len(x)
    correlate /= ((N - 1) * (np.std(x) * np.std(y)))
    M = len(correlate)
    return correlate[M/2 - t:M/2 + t]


def periodogram(signal, window, fs):
    signal = signal * window
    samples_count = len(signal)
    signal_fft = np.fft.fft(signal, samples_count) / np.sqrt(samples_count)
    power = signal_fft * signal_fft.conj() / np.sum(window ** 2)
    power = power.real  # Konwersja typów (zespolona -> rzeczywista), mimo, że P i tak ma wartości rzeczywiste)
    frequencies = np.fft.fftfreq(samples_count, 1.0 / fs)
    return np.fft.fftshift(power), np.fft.fftshift(frequencies)


def power_welch(s, w, shift, fs):
    signals_count = s.shape[0]
    samples_count = s.shape[1]
    window_length = len(w)
 
    segment_starts = np.arange(0, samples_count - window_length + 1, shift)
    segment_count = len(segment_starts)
    overlap = window_length * segment_count / float(samples_count)
    mean_power = np.zeros((signals_count, window_length))
    for i in range(signals_count):
        for j in range(segment_count):
            segment = s[i][segment_starts[j]:segment_starts[j] + window_length]
            power, frequencies = periodogram(segment, w, fs)
            mean_power[i] += power
    return mean_power / overlap, frequencies


def SFT(x, f, Fs):
    N = len(x)
    X = np.zeros(len(f), dtype = complex)
    i = 0
    for freq in f:
        X[i] = np.sum(x * np.exp(-2*np.pi*1j*(freq / Fs) * np.arange(N)))
        i += 1
    return X / np.sqrt(N)


def coherence(x, y, f, Fs):
    N = len(x)
    covarianceX = covariance(x, x, N)
    covarianceY = covariance(y, y, N)
    covarianceXY = covariance(x, y, N)
    Sx = SFT(covarianceX, f, Fs)
    Sy = SFT(covarianceY, f, Fs)
    Sxy = SFT(covarianceXY, f, Fs)
    phaseShift = np.angle(Sxy)

    transferTime = phaseShift / (2 * np.pi * f)
    Cxy = Sxy / (np.sqrt(Sx * Sy))

    return Cxy, transferTime


def coherence_fast(x, y, Fs):
    N = len(x)
    covarianceX = covariance_fast(x, x)
    covarianceY = covariance_fast(y, y)
    covarianceXY = covariance_fast(x, y)
    Sx = np.fft.fft(covarianceX)
    Sy = np.fft.fft(covarianceY)
    Sxy = np.fft.fft(covarianceXY)
    phaseShift = np.angle(Sxy)

    #transferTime = phaseShift / (2 * np.pi * f)
    Cxy = Sxy / (np.sqrt(Sx * Sy))

    return Cxy, phaseShift


def shuffle(array, axis = 0):
    shuffled = cp.deepcopy(array)
    for i in range(shuffled.shape[axis]):
        np.random.shuffle(shuffled[i])
    return shuffled


def find_extrema(signal, threshold_coefficient=0.262):
    threshold = threshold_coefficient * np.std(signal)
    signal[signal < threshold] = 0
    signal[signal > 0] = np.max(signal)
    signal_diff = np.diff(signal)
    signal_diff[signal_diff < 0] = 0

    relative_extrema = ss.argrelextrema(signal_diff, np.greater)[0]

    pauses = np.zeros((150, 2), dtype = int)

    idx = 0
    for i in range(1, len(relative_extrema)):
        if relative_extrema[i] - relative_extrema[i - 1] > 160:
            pauses[idx][1] = int(relative_extrema[i])  # koniec przedziału (bez swiecenia diody)
            pauses[idx][0] = relative_extrema[i - 1]  # początek przedziału bez świecenia
            idx += 1

    sections = np.zeros((150, 2), dtype=int)  #150 serii błysków, początek i koniec każdej
    for i in range(0, len(pauses)):
        sections[i][0] = int(pauses[i - 1][1])
        sections[i][1] = int(pauses[i][0])
    sections[0][0] = int(relative_extrema[0])  #poczatek swiecenia diody byl wczesniej niz 1. maksimum
    sections[-1][1] = int(relative_extrema[-1])  #koniec swiecenia diody byl pozniej niz ost. maksimum
    return sections[:, 0][sections[:, 0] != 0], sections[:, 1][sections[:, 1] != 0]


def get_frequencies(signal, slice_start, slice_length, fs):
    frequencies = []
    for i in range(len(slice_start)):
        signal_slice = signal[slice_start[i]:slice_start[i] + slice_length]
        slice_fft = np.fft.fftshift(np.fft.fft(signal_slice))
        frequencies_fft = np.fft.fftshift(np.fft.fftfreq(len(signal_slice), 1.0/fs))
        frequencies_fft = frequencies_fft[len(signal_slice)/2+1*10:len(signal_slice)/2+50 * 10]
        slice_fft = slice_fft[len(signal_slice)/2+1*10:len(signal_slice)/2+50 * 10]
        frequencies.append(int(round(frequencies_fft[np.argmin(slice_fft)])))
    return frequencies


def noise_level(signal, slice_start, slice_length, frequencies, f):
    if len(frequencies) != slice_start.shape[0]:
        print('Error!')
        return
    frequencies_count = slice_start.shape[0]
    slices_count = slice_start.shape[1]
    noise = np.zeros((frequencies_count - 1, slices_count, slice_length))
    for i in range(frequencies_count):
        if frequencies[i] != f:
            for j in range(slices_count):
                print(slice_length)
                print(signal[i][j][slice_start[i][j]:slice_start[i][j] + slice_length].shape)
                signal_slice = signal[i][j][slice_start[i][j]:slice_start[i][j] + slice_length]
                slice_fft = np.fft.fftshift(np.fft.fft(signal_slice))
                print(slice_fft.shape)
                noise[i][j] = slice_fft


def white_noise(std_dev, fs, T):
    """
    Generuje biały szum o podanych parametrach.
    :param std_dev: odchylenie standardowe,
    :param fs: częstotliwość próbkowania,
    :param T: czas trwania w sekundach.
    :return: t
    """

    t = np.linspace(0, T, fs * T, False)
    k = np.random.normal(loc=0, scale=std_dev, size=len(t))
    return t, k


def sin(f, fs, T, phi=0, A=1):
    """
    Generuje sygnał sinusoidalny o podanych parametrach.
    :param f: częstotliwość funkcji,
    :param fs: częstotliwość próbkowania,
    :param T: czas trwania,
    :param phi: przesunięcie fazowe,
    :param A: ampliduda
    :return: t - wektor czasu, s - wektor próbek
    """
    t = np.linspace(0, T, fs * T, endpoint=False)
    s = A * np.sin(2 * np.pi * t * f + phi)
    return t, s


def akaike(data, k, N, p):
    """
    Wyznacza współczynniki rząd i współczynniki modelu na podstawie kryterium Akaikego.
    :param data: dane,
    :param k: liczba kanałów,
    :param N: liczba próbek,
    :param p: rząd modelu,
    :return:
    """
    p_akaike = []
    akaike = []
    if k == 1:
        data = np.reshape(data, (1, -1))
    for i in range(p + 1):
        _, V = mult_AR(data, i, 1)
        akaike.append(np.log(np.linalg.det(V)) + 2 * i * k * k / N)
        p_akaike.append(i)
    return p_akaike, akaike


def z_transform(data, p, fs, f0, fmax):
    """
    Wyznacza transformatę Z z danych na podstawie podanego rzędu modelu dla podanych częstotliwości.
    :param data: dane,
    :param p: rząd modelu,
    :param fs: częstotliwość próbkowania,
    :param f0: początkowa częstotliwość dla której ma być wyznaczona transformata,
    :param fmax: końcowa częstotliwość dla której ma być wyznaczona transformata.
    :return:
    """
    if len(np.shape(data)) == 1:
        data = np.reshape(data, (1, -1))
    k, N = np.shape(data)
    A, V = mult_AR(data, p, 1)
    A = -A
    freqs = np.arange(f0, fmax)  # wektor częstości
    Af = np.zeros((fmax - f0, k, k), 'complex')  # A(f)
    Hf = np.zeros((fmax - f0, k, k), 'complex')  # H(f)
    S = np.zeros((fmax - f0, k, k), 'complex')  # widma
    for f in range(f0, fmax):
        z = np.exp(2 * np.pi * 1j * f * (1. / fs))
        Az = np.eye(k, dtype='complex')  # żeby A(0) to były jedynki na diagonali
        for pa in range(p):
            Az += A[pa] * z ** (-pa - 1)
        Af[f] = Az
        Hf[f] = np.linalg.inv(Az)
        h = np.matrix(Hf[f])
        S[f] = h * np.matrix(V) * h.getH()  # getH  Returns the (complex) conjugate transpose of self.
    return freqs, Af, Hf, S


def partial_coherences(S, k, f0, fmax):
    """
    Wyznacza koherencje cząstkowe.
    :param S: macierz widm po transformacie z
    :param k:
    :param f0:
    :param fmax:
    :return:
    """
    C = np.zeros((fmax - f0, k, k), 'complex')  # widma
    for f in range(f0, fmax):
        d = np.linalg.inv(S[f])
        for i in range(k):
            for j in range(k):
                C[f, i, j] = ((-1) ** (i + j)) * d[j, i] / ((d[i, i] * d[j, j]) ** 0.5)
    return C


def ordinary_coherences(S, k, f0, fmax):
    """
    Wyznacza koherencje zwyczajne
    :param S: macierz widm po transformacie z
    :param k:
    :param f0:
    :param fmax:
    :return:
    """
    K = np.zeros((fmax - f0, k, k), 'complex')  # widma
    for f in range(f0, fmax):
        for i in range(k):
            for j in range(k):
                K[f, i, j] = S[f, i, j] / ((S[f, i, i] * S[f, j, j]) ** 0.5)
    return K


def hjotr_transform(signal_data_info):
    """
    Zwraca transformatę Hjorta dla kanałów C3, Cz oraz C4.
    :param signal_data_info: obiekt typu SignalDataInfor z sygnałami i informacjami o nich,
    :return: C3H, CzH, C4H - transformaty Hjorta dla kanałów C3, Cz, C4.
    """

    c3_neighbours = ['T3', 'F3', 'Cz', 'P3']
    cz_neighbours = ['C3', 'Fz', 'C4', 'Pz']
    c4_neighbours = ['Cz', 'F4', 'T4', 'P4']
    C3H = signal_data_info['C3'] - 0.25 * (np.sum([signal_data_info[i] for i in c3_neighbours]))
    CzH = signal_data_info['Cz'] - 0.25 * (np.sum([signal_data_info[i] for i in cz_neighbours]))
    C4H = signal_data_info['C4'] - 0.25 * (np.sum([signal_data_info[i] for i in c4_neighbours]))
    return C3H, CzH, C4H
