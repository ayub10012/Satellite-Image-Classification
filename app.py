from flask import Flask,render_template, redirect, request
import numpy as np
import PIL
from tensorflow.keras.models import load_model
import cv2

model = load_model('model/satellite.h5')

lis = ['Cloudy/Cant determine/Snow','Dessert','Grasslands/Moutains/land with greenery','Water']


def est2(test_image,model,labels):
    img = cv2.imread(test_image)
    img = img / 255.0
    img = cv2.resize(img,(224,224))
    img = img.reshape(1,224,224,3)
    prediction = model.predict(img)
    pred_class = np.argmax(prediction, axis = -1)
    return labels[pred_class[0]]


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')


#render home page and then we have classification fundtion 
# classify the image and all that is there






@app.route('/ALL')
def all():
    return render_template('detection.html')
@app.route("/submit2", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image2']

		img_path = "./static/" + img.filename	
		img.save(img_path)

		p = est2(img_path, model ,lis)

	return render_template("detection.html", prediction = p, img_path = img_path)





if __name__ == '__main__':
    app.run()