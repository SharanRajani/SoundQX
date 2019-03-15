import sendgrid
from flask import request, render_template, flash, current_app, jsonify
from application import app, logger, cache
from application.decorators import threaded_async
from application.models import *
from application import forms
#from forms import *
from time import time
import glob
import pandas as pd
import os
import sys
import tkinter
from  tkinter import *
from tkinter import filedialog
import matplotlib.mlab
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
import soundfile as sf
import cv2
from keras.models import load_model
from flask import redirect
import os
from werkzeug.utils import secure_filename
basedir = os.path.abspath(os.path.dirname(__file__))

uploaded_filename = ''


# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



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

    
def predict(wavfile,specpng):
    model = load_model("/home/thegodfather/Desktop/A2IoT/Industrial-Processes-Quality-Assessment-using-Sound-Analytics/Flask-Easy-Template/resources/alex-cnn.h5")
    plotstft(wavfile,specpng)

    img = cv2.imread(specpng)
    img = cv2.resize(img, (224,224))
    img = np.reshape(img, (1,224,224,3))
    # print(img.shape)
    pred = model.predict(img)
    # print(pred)
    if(pred[0][0]>0.66):
        output="Defective"
    elif(pred[0][0]<0.33):
        output="Excellent"
    else:
        output="Satisfactory"
    return output



@app.route('/')
@app.route('/index')
@app.route('/index/<int:page>')
def index(page=1):
    m_tasks = SampleTable()

    list_records = m_tasks.list_all(page, app.config['LISTINGS_PER_PAGE'])

    return render_template("index.html", list_records=list_records)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        global uploaded_filename
        uploaded_filename = os.path.join(basedir, app.config['UPLOAD_FOLDER'], filename)
        file.save(uploaded_filename)
        return redirect("http://localhost:5000/add_record")
    return "Error" 

from flask import send_from_directory
@app.route('/add_record', methods=['GET', 'POST'])
def open():
    if request.method == 'GET':
        output=predict(uploaded_filename,"/home/thegodfather/Desktop/A2IoT/for_interns/Test/three.png")
        return render_template("add_record.html",qclass=output)

@app.route('/', methods=['POST'])
def my_form_post():
    text = request.form['text']
    processed_text = text.upper()
    return processed_text


@threaded_async
def send_email(app, to, subject, body):
    with app.app_context():
        sg = sendgrid.SendGridClient("SG.pRFA8c9bRXXXXXXXXXXXXXXXXXXXXXXXXX")
        message = sendgrid.Mail()
        message.add_to(to)
        message.set_subject(subject)
        message.set_html(body)
        message.set_from('Template No-Reply <noreplay@flaskeasytemplate.com>')
        try:
            status, msg = sg.send(message)
            print("Status: " + str(status) + " Message: " + str(msg))
            if status == 200:
                return True
        except Exception as ex:
            print("------------ ERROR SENDING EMAIL ------------" + str(ex.message))
    return False


@app.route('/contact', methods=['GET', 'POST'])
def contact():
    recaptcha = current_app.config['RECAPTCHA_SITE_KEY']
    email_sent = False

    if request.method == 'POST':
        email = request.form['email']
        name = request.form['name']
        message = request.form['message']
        recaptcha_response = request.form['g-recaptcha-response']

        send_email(app, to=current_app.config['ADMIN_EMAIL'], subject="Contact Form Flask Shop",
                   body=email + " " + name + " " + message)

        email_sent = True

    return render_template("contact.html", RECAPTCHA_SITE_KEY=recaptcha, email_sent=email_sent)


# ----- UTILS. Delete them if you don't plan to use them -----

@app.route('/cache_true')
@cache.cached(timeout=120)
def cached_examples():
    start = time()
    records = SampleTable().benchmark_searchspeed()
    return jsonify(data=records, cached_at=datetime.datetime.now(), done_in=time() - start)


@app.route('/cache_false')
def not_cached_examples():
    start = time()
    records = SampleTable().benchmark_searchspeed()
    return jsonify(result=records, cached_at=datetime.datetime.now(), done_in=time() - start)





@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500
