import cv2
import tensorflow as tf
from tensorflow import keras
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)
path = r'C:\Users\a\Downloads\FaceMask3'
model = keras.models.load_model(path)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

cap = cv2.VideoCapture(0)

while True:
    keypress = cv2.waitKey(1)
    if keypress & 0xFF == ord('q'):
        break

    ret, img = cap.read()

    img = cv2.resize(img, IMG_SIZE)
    cv2.imshow('t', img)

    im = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = im.astype('float32')

    im = tf.expand_dims(im, 0)
    predictions = model.predict(im)
    score = predictions[0][0]
    percent_mask = (1-score)*100
    print(f'{percent_mask:.2f}% mask')

    
    


cap.release()
cv2.destroyAllWindows()
