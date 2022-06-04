import os
from flask import Flask, request, redirect, url_for, flash, render_template
from image_prediction import predict_image
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/images'

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def homepage():
    return render_template("index.html")

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Upload a file')
            return redirect(request.url)
        if file:
            identification_type = ""
            if request.form.get('crop-identification'):
              identification_type = "crop"
            if request.form.get('disease-identification'):
              identification_type = "disease"
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploadedImageFilePath = os.path.join(
                app.config['UPLOAD_FOLDER'], filename)
            fixedPath = os.path.join('images/', filename)
            result = predict_image(uploadedImageFilePath, identification_type)
            return redirect(url_for('image_analysis', path=fixedPath, result=result))
    return render_template("index.html")

@app.route('/image', methods=['GET'])
def image_analysis():
    path = request.args.get('path')
    result = request.args.get('result')
    return render_template("image.html", path=path, result=result)

if __name__ == "__main__":
    app.run()
