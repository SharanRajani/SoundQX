from flask import Flask, render_template, request
from werkzeug import secure_filename
import test_gen_spec
from flask import session, redirect
import spec_plot
import cv2
from keras.models import load_model
import numpy as np
import scipy.io.wavfile as wav
import librosa


app = Flask(__name__)
app.secret_key = "my precious"

@app.route('/', methods = ['GET', 'POST'])
def home():
	f = 'wav/landing_page.wav'
	if request.method == 'POST':
		f = request.files['file']
		print(f.filename)
		y,sr=librosa.load('../chunks_test/'+secure_filename(f.filename),sr=48000)
		print(type(y))
		y = 10000000* y 
		ampedpath = "./static/wav/amped.wav"
		wav.write(ampedpath,48000,y)
		f.save('./static/wav/'+secure_filename(f.filename))
		f = 'wav/'+f.filename
		session['filepath'] = './static/' + f
		return render_template('First.html', wav_file = f)
	return render_template('First.html', wav_file = f)


@app.route('/display_spec')
def display_spec():
	filepath = session['filepath']
	modelpath = "model.hdf5"
	with open("./static/wav/temp", "w") as file:
	        file.write(filepath)

	file.close()

	mixpath = "./static/wav/temp" 

	enhancedpath = test_gen_spec.predict(modelpath, mixpath)

	spec_plot.plotstft(enhancedpath, "./static/images/enhanced_spectograms.png")
	spec_plot.plotstft(filepath, "./static/images/original_spectograms.png")

	return render_template('Second.html', wav_file = filepath)

@app.route('/classify')
def classify():
	enhancedpng = "./static/images/enhanced_spectogram.png"

	model = load_model("./alex-cnn.h5")

	img = cv2.imread(enhancedpng)
	img = cv2.resize(img, (224,224))
	img = np.reshape(img, (1,224,224,3))
	print(img.shape)
	pred = model.predict(img)

	return render_template('Third.html', pred = pred)


if __name__ == '__main__':
	app.run(debug=True)
