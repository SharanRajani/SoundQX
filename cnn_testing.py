import cv2
from keras.models import load_model
import numpy as np
print("IM IN CNN TESTING")
def predict(filename):
	model = load_model("./alex-cnn.h5")
	# filename = "/home/atharva/a2iot/deeplearning/DDAE/spectrograms/spectrograms_chunks_noise/chunk24.png"
	# filename = "/home/atharva/a2iot/deeplearning/DDAE/spectrograms/spectrograms_test_enhanced/chunk1107.png"

	img = cv2.imread(filename)
	img = cv2.resize(img, (224,224))
	img = np.reshape(img, (1,224,224,3))
	print(img.shape)
	pred = model.predict(img)
	print(pred)
	return pred