import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import random as python_random
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

np.random.seed(10)
python_random.seed(10)
tf.random.set_seed(10)

model = tf.keras.models.load_model('models/diseases_ft.hdf5') 

dataset_path = 'datasets\\diseases\\test'

sub_folders = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name))]

test_data_gen = ImageDataGenerator(rescale=1./255.0, preprocessing_function=preprocess_input)

test_set = test_data_gen.flow_from_directory(dataset_path,
target_size=(224, 224),
class_mode=None,
classes=sub_folders,
batch_size=1,
shuffle=False,
seed=10)

true_classes = test_set.classes
class_indices = test_set.class_indices
class_indices = dict((v,k) for k,v in class_indices.items())

vgg_preds = model.predict(test_set)
vgg_pred_classes = np.argmax(vgg_preds, axis=1)

vgg_acc = accuracy_score(true_classes, vgg_pred_classes)
print("Accuracy: {:.2f}%".format(vgg_acc * 100))

print(confusion_matrix(true_classes, vgg_pred_classes))

print(classification_report(true_classes, vgg_pred_classes))