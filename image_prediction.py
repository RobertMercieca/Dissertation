from allinone import preprocess_image
import tensorflow as tf
import numpy as np
import os
import tensorflow as tf
import cv2

crops_class_names = ['apple', 'grape', 'potato']

diseases_class_names = ['apple scab', 'apple cedar rust', 'apple healthy', 'grape black rot', 'grape black measles', 'grape healthy',  'potato early blight', 'potato healthy', 'potato late blight']

crop_model = tf.keras.models.load_model('models/crops.hdf5')

disease_model = tf.keras.models.load_model('models/diseases_ft.hdf5') 

def predict_image(path, identification_type):
    path_new = ""
    if (identification_type == "crop"):
      path_new = preprocess_image(path)
    else:
      path_new = path

    im_bgr = cv2.imread(path_new)
    im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
    im_rgb = cv2.resize(im_rgb,(224,224))
    im_rgb = im_rgb.reshape(1,224,224,3)
    im_arr = np.array(im_rgb)
    im_arr = tf.keras.applications.vgg16.preprocess_input(im_arr)
    im_arr = im_arr/255.0
    image_to_predict = im_arr

    if (identification_type == "crop"):
      os.remove('final.jpg')

    result = ""
    
    if (identification_type == "crop"):
      result = crop_model.predict(image_to_predict, batch_size=1)

      predicted_crop = crops_class_names[np.argmax(result)]
      certainty_value = 100 * np.max(result)

      
      result = "This image is a {}, certainty level: {:.2f}%.".format(
                  predicted_crop, certainty_value)

    else:
      result = disease_model.predict(image_to_predict, batch_size=1)

      predicted_disease = diseases_class_names[np.argmax(result)]
      certainty_value = 100 * np.max(result)

      
      result = "This image is a {}, certainty level: {:.2f}%. <br>".format(
                  predicted_disease, certainty_value)

    return result

