from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
	f = 'wav/no_tree_ent.wav'
	if request.method == 'POST':
		f = request.files['file']
		print(f.filename)
		f.save('./static/wav/'+secure_filename(f.filename))
		f = 'wav/'+f.filename
	return render_template('one.html', wav_file = f)

@app.route('/flash_spec')
def display_spec():
	return render_template('two (copy).html')

if __name__ == '__main__':
	app.run(debug=True)
