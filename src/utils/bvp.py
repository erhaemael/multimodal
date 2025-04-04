from biosppy import bvp
import numpy as np
import peakutils
import math

def bvpPeaks(signal):
    """Menemukan puncak dalam sinyal BVP."""
    cb = np.array(signal)
    x = peakutils.indexes(cb, thres=0.02 / max(cb), min_dist=0.1)
    y = []
    
    i = 0
    while i < len(x) - 1:
        if x[i + 1] - x[i] < 15:
            y.append(x[i])
            x = np.delete(x, i + 1)
        else:
            y.append(x[i])
        i += 1
    return y

def getRRI(signal, sample_rate):
    """Menghitung interval RR (RRI) dari sinyal BVP."""
    peakIDX = bvpPeaks(signal)
    spr = 1 / sample_rate
    ibi = [0, 0]
    
    for i in range(1, len(peakIDX)):
        ibi.append((peakIDX[i] - peakIDX[i - 1]) * spr)
    return ibi

def getHRV(ibi, avg_heart_rate, method='SDNN'):
    """Menghitung HRV menggunakan metode SDNN atau RMSSD."""
    rri = np.array(ibi) * 1000
    RR_list = rri.tolist()
    RR_diff, RR_sqdiff = [], []
    cnt = 2

    while cnt < (len(RR_list) - 1):
        RR_diff.append(abs(RR_list[cnt + 1] - RR_list[cnt]))
        RR_sqdiff.append(math.pow(RR_list[cnt + 1] - RR_list[cnt], 2))
        cnt += 1

    hrv_window_length = 10
    window_length_samples = int(hrv_window_length * (avg_heart_rate / 60))
    SDNN, RMSSD = [], []
    index = 1

    for val in RR_sqdiff:
        if index < int(window_length_samples):
            SDNNchunk = RR_diff[:index]
            RMSSDchunk = RR_sqdiff[:index]
        else:
            SDNNchunk = RR_diff[(index - window_length_samples):index]
            RMSSDchunk = RR_sqdiff[(index - window_length_samples):index]
        
        SDNN.append(np.std(SDNNchunk))
        RMSSD.append(math.sqrt(1. / len(RR_list) * np.std(RMSSDchunk)))
        index += 1

    SDNN, RMSSD = np.array(SDNN, dtype=np.float32), np.array(RMSSD, dtype=np.float32)
    return SDNN if method == 'SDNN' else RMSSD

def extract_features(signal, sample_rate=32):
    """Ekstrak fitur HR & HRV dari sinyal BVP."""
    _, filtered_signal, _, _, hr = bvp.bvp(signal.flatten(), sample_rate, show=False)
    hr = np.mean(hr)
    ibi = getRRI(filtered_signal, sample_rate)
    hrv = getHRV(ibi, hr, method='RMSSD')

    return {"HR_BVP": hr, "HRV_BVP": hrv.mean()}
