from flask import Flask, render_template, request
from werkzeug import secure_filename
import test_gen_spec
from flask import session


app = Flask(__name__)
app.secret_key = "my precious"

@app.route('/', methods = ['GET', 'POST'])
def home():
	f = 'wav/landing_page.wav'
	if request.method == 'POST':
		f = request.files['file']
		print(f.filename)
		f.save('./static/wav/'+secure_filename(f.filename))
		f = 'wav/'+f.filename
	return render_template('First.html', wav_file = f)


@app.route('/flash_spec')
def display_spec():
	filepath = session['filepath']
	modelpath = "model.hdf5"
	with open("./static/wav/temp", "w") as file:
	        file.write(filename)

	file.close()

	mixpath = "./static/wav/temp" 

	enhancedpath = test_gen_spec.predict(modelpath, mixpath)

	spec_plot.plotstft(enhancedpath, "./static/images/enhanced_spectogram.png")
	spec_plot.plotstft(filepath, "./static/images/original_spectogram.png")

	return render_template('two (copy).html', wav_file = filepath)

if __name__ == '__main__':
	app.run(debug=True)
