from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

IMG_SIZE = 200

# import tensorflow

# print(tensorflow.__version__)

model = load_model('C:/Learning/deploy-ml_2/app/my_modelResnet.h5')

# model.summary()

img = load_img('C:/Books/Programming/ML/Francois Chollet - Deep Learning with Python - 2017/projects/catsvsdogs/test1/test1/18.jpg', target_size=(IMG_SIZE, IMG_SIZE))
img_arr = img_to_array(img)
# print(img_arr)

predictions = np.argmax(model.predict([[img_arr]]),axis=1)
print(predictions)