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
    session['filepath']=None
    f = 'wav/landing_page.wav'
    if request.method == 'POST':
        f = request.files['file']
        print(f.filename)
        f.save('./static/wav/'+secure_filename(f.filename))
        f = 'wav/'+f.filename
        session['filepath'] = './static/' + f
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

	spec_plot.plotstft(enhancedpath, "./static/images/enhanced_spectogram.png")
	spec_plot.plotstft(filepath, "./static/images/original_spectogram.png")

	return render_template('Second.html', wav_file = filepath)

@app.route('/classify')
def classify():
    #session['filepath'] =  None
    model = load_model("./static/alex-cnn.h5")
    filename = "./static/images/enhanced_spectogram.png"
    # filename = "/home/atharva/a2iot/deeplearning/DDAE/spectrograms/spectrograms_test_enhanced/chunk1107.png"
    img = cv2.imread(filename)
    img = cv2.resize(img, (224,224))
    img = np.reshape(img, (1,224,224,3))
    print(img.shape)
    pred = model.predict(img)[0][1]
    #pred= 0.5
    print(pred)
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

if __name__ == '__main__':
	app.run(debug=True)
