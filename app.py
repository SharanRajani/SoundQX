from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask(__name__)

@app.route('/', methods = ['GET', 'POST'])
def home():
	if request.method == 'POST':
		f = request.files['file']
		f.save('./static/wav/'+secure_filename(f.filename))
		print(f)
	return render_template('one.html')

if __name__ == '__main__':
	app.run(debug=True)