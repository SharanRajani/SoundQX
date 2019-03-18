import glob
import pandas as pd
import os
import sys

#from  Tkinter import *
import tkinter
from  tkinter import *
from tkinter import filedialog

import matplotlib.mlab

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import soundfile as sf


def select_dir():
    root = tkinter.Tk()
    print("Select the folder containing .wav files to create spectrograms")
    root.directory = filedialog.askdirectory()
    print (root.directory)
    wav_path = root.directory + "/"
    original_dir_name = root.directory.split("/")[-1]
    png_path = root.directory.rsplit("/",1)[0] + "/"+ "spectrograms" + "/" + "spectrograms_" + original_dir_name + "/"

    print("wav_path: ")
    print(wav_path)

    print("original_dir_name: ")
    print(original_dir_name)

    print("png_path: ")
    print(png_path)
    return wav_path, original_dir_name, png_path

def create_spectrograms():

    wav_path, original_dir_name, png_path = select_dir()
    wavfiles = glob.glob(os.path.join(wav_path+"*.wav"))
    count = 0

    for audiofile in wavfiles:
        png_name = audiofile.split("/")[-1].rsplit(".",1)[0]
        png_final_path = png_path + png_name + ".png"

        if os.path.exists(png_path):
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
        else:
            os.makedirs(png_path)
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
    return count, png_path, wav_path

def create_spectrograms_internal(wav_path,png_path):

    #wav_path, original_dir_name, png_path = select_dir()
    wavfiles = glob.glob(os.path.join(wav_path+"*.wav"))
    count = 0

    for audiofile in wavfiles:
        png_name = audiofile.split("/")[-1].rsplit(".",1)[0]
        #png_path = png_path + "/" + "spectrograms" + "/"
        png_final_path = png_path + png_name + ".png"
        print(png_final_path)

        if os.path.exists(png_path):
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
        else:
            os.makedirs(png_path)
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
    return count  



""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))
    
    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)   
    # cols for windowing
    cols = int(np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1)
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))
    
    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win
    
    return np.fft.rfft(frames)    
    
""" scale frequency axis logarithmically """
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))
    scale = scale.astype(int)
    
    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,scale[i]:], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,scale[i]:scale[i+1]], axis=1)
    
    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]
    
    return newspec, freqs


""" plot spectrogram"""

def plotstft(audiopath, plotpath=None, binsize=2**10, colormap="jet"):
    samples, samplerate = sf.read(audiopath)
    s = stft(samples, binsize)
    
    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    
    timebins, freqbins = np.shape(ims)
    
    #plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower", aspect="auto", cmap=colormap, interpolation="none")
    #plt.tick_params(direction='out', length=6, width=2, colors='r',grid_color='r', grid_alpha=0.5)
    plt.tick_params(axis='both', which='both', bottom='off',top='off',left='off',right='off',labelbottom='off',labelleft='off')
    #plt.colorbar()
    #plt.axis('off')

    fig = plt.gcf()

    #plt.xlabel("time (s)")
    #plt.ylabel("frequency (hz)")
    #plt.xlim([0, timebins-1])
    #plt.ylim([0, freqbins])

    #xlocs = np.float32(np.linspace(0, timebins-1, 5))
    #plt.xticks(xlocs, ["%.02f" % l for l in ((xlocs*len(samples)/timebins)+(0.5*binsize))/samplerate])
    #ylocs = np.int16(np.round(np.linspace(0, freqbins-1, 10)))
    #plt.yticks(ylocs, ["%.02f" % freq[i] for i in ylocs])
    
    fig.savefig(plotpath, bbox_inches="tight", pad_inches=0)
    #else:
        #plt.show()
        
    plt.clf()
    plt.close()


#plotstft("/home/aratrika/aratrika_work/data sets/UrbanSound_ex/test_wav_png/7061-6-0-0.wav")
#count = create_spectrograms()
def iterate_folds_spec():
    wav_path_list=[]
    png_path_list=[]
    user_input = 'c'
    while user_input == 'c':       
        #count = create_spectrograms()
        wav_path, original_dir_name, png_path = select_dir()
        wav_path_list.append(wav_path)
        png_path_list.append(png_path)
        print('Press c to continue selecting paths: \n')
        user_input = raw_input()
    length = int(len(wav_path_list))
    for j in range(length):
        print(wav_path_list[j])
        print(png_path_list[j])
    
    for i in range(length):
        create_spectrograms_internal(wav_path_list[i],png_path_list[i])

def copy_spec(): #Copy spectrograms from one folder to another
    user_input = 'c'
    while(user_input == 'c'):
        root1 = tikinter.Tk()
        print("Select the folder containing spectrograms: ")
        root1.directory = filedialog.askdirectory()
        print (root1.directory)
        selected_path = root1.directory + "/"

        print("Select the folder to be copied to: ")
        root2 = tkinter.Tk()
        root2.directory = filedialog.askdirectory()
        print (root2.directory)
        final_path = root2.directory + "/"

        specfiles = glob.glob(os.path.join(selected_path+"*.png"))

        for spectrograms in specfiles:
            if os.path.exists(final_path):
                shutil.copy2(spectrograms,final_path)
            else:
                os.makedirs(final_path)
                shutil.copy2(spectrograms,final_path)
        print('Enter c to continue')
        user_input = raw_input()



def see_spec(wav_path,png_path):

    #wav_path, original_dir_name, png_path = select_dir()
    wavfiles = glob.glob(os.path.join(wav_path+"*.wav"))
    count = 0

    for audiofile in wavfiles:
        png_name = audiofile.split("/")[-1].rsplit(".",1)[0]
        #png_path = png_path + "/" + "spectrograms" + "/"
        png_final_path = png_path + png_name + ".png"
        print(png_final_path)

        if os.path.exists(png_path):
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
        else:
            os.makedirs(png_path)
            plotstft(audiofile,png_final_path)
            count +=1
            print(count)
    return count




# create_spectrograms()