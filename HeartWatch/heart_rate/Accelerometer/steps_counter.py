from numpy.linalg.linalg import norm
import pandas as pd
from scipy.ndimage.measurements import label
pd.__version__
import matplotlib as mpl
import numpy as np
mpl.rc('font', size=20)
mpl.rc('figure', figsize=(20, 10))
import warnings
warnings.filterwarnings("ignore")
from bitstring import BitStream, BitArray
from scipy.signal import butter, lfilter
import scipy.signal as signal
import math
from detecta import detect_peaks
import matplotlib.pyplot as plt




def DecimalToBinary(num):

  if num >= 0:
    dec=bin(num).lstrip('0b')
    
  if num<0:
    dec=num.to_bytes(1,'big',signed=True)
    dec=BitArray(bytes=dec).bin
  if len(dec)<8:
    diff=8-len(dec)
    for i in range(diff):
      dec='0'+dec
    # print(dec)
  return dec
def as_signed_big(binary_str):
    # This time, taking advantage of positional args and default values.
    as_bytes = int(binary_str, 2).to_bytes(2, 'big')
    return int.from_bytes(as_bytes, 'big', signed=True)

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y
def normalize(x):
    return x/np.max(np.abs(x))
def scale(X):
    return (X-np.mean(X))/np.std(X)

def low_pass_IIR(data,fl,samp_f,order):
    b, a = signal.butter(order, fl/(samp_f/2), btype='low', output='ba')
    low_data = signal.lfilter(b, a, data)
    return low_data

def high_pass_IIR(data,fh,samp_f,order):
    b, a = signal.butter(order, fh/(samp_f/2), btype='high', output='ba')
    high_data = signal.lfilter(b, a, data)
    return high_data


fs=20
n=30*20

def to_decimal(acc):


  acc_Xind=[]
  for j in range(acc.shape[0]):
      for i in range(1,121,6):    
          acc_Xind.append(DecimalToBinary(acc.iloc[j,i+1].item())[-1::-1]+DecimalToBinary(acc.iloc[j,i].item())[-1::-1])

  acc_X=[]
  for i in range(len(acc_Xind)):
      acc_X.append(as_signed_big(acc_Xind[i]))
  acc_X=np.asarray(acc_X)

  acc_Yind=[]
  for j in range(acc.shape[0]):
      for i in range(3,121,6):    
          acc_Yind.append(DecimalToBinary(acc.iloc[j,i+1].item())[-1::-1]+DecimalToBinary(acc.iloc[j,i].item())[-1::-1])

  acc_Y=[]
  for i in range(len(acc_Yind)):
      acc_Y.append(as_signed_big(acc_Yind[i]))
  acc_Y=np.asarray(acc_Y)

  acc_Zind=[]
  for j in range(acc.shape[0]):
      for i in range(5,121,6):    
          acc_Zind.append(DecimalToBinary(acc.iloc[j,i+1].item())[-1::-1]+DecimalToBinary(acc.iloc[j,i].item())[-1::-1])

  acc_Z=[]
  for i in range(len(acc_Zind)):
      acc_Z.append(as_signed_big(acc_Zind[i]))
  acc_Z=np.asarray(acc_Z)

  return acc_X , acc_Y , acc_Z


def step_count(acc_X , acc_Y, acc_Z ): 
    steps_1_4win=[]
    time_stamp=[]
    peaks_all=[]
    a_x = acc_X*18.3/128.0/1000.0+0.06
    a_gx=low_pass_IIR(a_x,0.19,fs,3)
    a_ux=a_x-a_gx
    a_y = acc_Y*18.3/128.0/1000.0+0.06
    a_gy=low_pass_IIR(a_y,0.19,fs,3)
    a_uy=a_y-a_gy
    a_z = acc_Z*18.3/128.0/1000.0+0.06
    a_gz=low_pass_IIR(a_z,0.19,fs,3)
    a_uz=a_z-a_gz

    a=(a_ux*a_gx)+(a_uy*a_gy)+(a_uz*a_gz)
    a_lp=low_pass_IIR(a,5,fs,3)
    a_lp_hp=high_pass_IIR(a_lp,1,fs,3)
    a_lp_hp=normalize(a_lp_hp)
    a_lp_hp=a_lp_hp-np.mean(a_lp_hp)

    peaks = detect_peaks(a_lp_hp,mph=0.2,mpd=8)
    valleys = detect_peaks(-a_lp_hp,mph=0.2,mpd=8)
    range_step = min(len(peaks) , len(valleys))
    step_length = []
    distance = 0
    for l in range(0, range_step):    
      length  = ((abs(peaks[l] - valleys[l]))**(1/float(4)))/2
      step_length.append(length)
    # print(step_length)  
    
    # plt.plot(a_lp_hp)
    # plt.scatter(peaks , a_lp_hp[peaks])
    # plt.scatter(valleys , a_lp_hp[valleys])

    # plt.show()
    steps_1_4win.append(len(np.unique(np.asarray(peaks))))
    # print(steps_1_4win)
    distance = np.average(step_length) * steps_1_4win[0]
    # print(distance)
    return steps_1_4win , distance

    






