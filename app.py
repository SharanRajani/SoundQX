from flask import Flask, render_template, request
from werkzeug import secure_filename
import test_gen_spec
from flask import session, redirect
import spec_plot
import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa
import cnn_testing
import createAmpGraph

app = Flask(__name__)
app.secret_key = "my precious"
enhancedpath = ""

session_filepath = ""

@app.route('/', methods = ['GET', 'POST'])
def home():
	# session['filepath']=None
	f = 'wav/chunk334.wav'
	if request.method == 'POST':
		f = request.files['file']
		print(f.filename)
		f.save('./static/wav/' + secure_filename(f.filename))
		f = 'wav/'+f.filename
		# session['filepath'] = './static/' + f
		global session_filepath
		session_filepath = './static/' + f
	return render_template('First.html', wav_file = f)


@app.route('/denoise_details', methods = ['GET', 'POST'])
def denoise_details():
	sample_rate, wav_data = wav.read(session_filepath)
	createAmpGraph.makegraph(wav_data)
	return render_template('New_Second.html',sample_rate=sample_rate,wav_data=wav_data)


@app.route('/display_spec')
def display_spec():
    filepath = session_filepath
    modelpath = "model.hdf5"
    with open("./static/wav/temp", "w") as file:
        file.write(filepath)
    file.close()
    mixpath = "./static/wav/temp"
    global enhancedpath
    enhancedpath = test_gen_spec.predict(modelpath, mixpath)
    spec_plot.plotstft(enhancedpath, "./static/images/enhanced_spectogram.png", "jet" )
    spec_plot.plotstft(filepath, "./static/images/original_spectogram.png", "jet")
    spec_plot.plotstft(enhancedpath, "./static/images/enhanced_spectogram_html.png", "PuBuGn")
    spec_plot.plotstft(filepath, "./static/images/original_spectogram_html.png", "PuBuGn")
    filepath=filepath[9:]
    enhancedpath = enhancedpath[7:]
    return render_template('Second.html', wav_file = filepath, wav_file_enhance = enhancedpath)

@app.route('/classify_details', methods = ['GET', 'POST'])
def classify_details():
	sample_rate, wav_data = wav.read("./static/wav/enhanced.wav")
	createAmpGraph.makegraph1(wav_data)
	return render_template('New_Third.html',sample_rate=sample_rate,wav_data=wav_data)

@app.route('/classify')
def classify():
	filename = "./static/images/original_spectogram.png"
	pred = cnn_testing.predict(filename)
	pred = pred[0][1]

	if(pred>0.75):
		pred_label = "Excellent!"
		pred_desc = "Congratuations! The quality of your weld is first-rate."
		pred_img_path = "./static/images/positive_result.png"
	elif(pred<0.25):
		pred_label = "Defective"
		pred_desc = "The quality of the weld is substandard."
		pred_img_path = "./static/images/negative_result.png"
	else:
		pred_label = "Satisfactory"
		pred_desc = "There is room for improvement. Please ensure weld quality does not deteriorate any further."
		pred_img_path = "./static/images/neutral_result.png"
	return render_template('Third.html', pred_label = pred_label, pred_desc = pred_desc, pred_img_path = pred_img_path)

@app.route('/validate', methods = ['GET', 'POST'])
def validate():
    filepath = session_filepath
    print(filepath)
    print(enhancedpath)
    return render_template('Sixth.html', wav_file = enhancedpath)

if __name__ == '__main__':
	app.run(debug=True, use_reloader=True)
