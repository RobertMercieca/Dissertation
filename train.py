import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, Flatten
import os
import numpy as np
import matplotlib.pyplot as plt
import random as python_random

np.random.seed(10)
python_random.seed(10)
tf.random.set_seed(10)

#crop or disease
modelfor = "disease"
modelname = "diseases_ft.hdf5"
finetune = "yes"

train_data_gen = ImageDataGenerator(
    rescale=1./255.0,
    rotation_range=90, 
    width_shift_range=0.3, 
    height_shift_range=0.3,
    vertical_flip=True,
    horizontal_flip=True, 
    brightness_range=[0.3, 0.7],
    validation_split=0.15,
    preprocessing_function=preprocess_input
  )

train_dataset_path = ""
dropout_value = 0
if (modelfor == "crop"):
  train_dataset_path = "datasets\\crops\\train"
  dropout_value = 0.2
else:
  train_dataset_path = "datasets\\diseases\\train"
  dropout_value = 0.55

sub_folders = [name for name in os.listdir(train_dataset_path) if os.path.isdir(os.path.join(train_dataset_path, name))]

num_classes = len(sub_folders)

class_names = sub_folders

batch_size = 64

train_set = train_data_gen.flow_from_directory(train_dataset_path,
  target_size=(224, 224),
  class_mode="categorical",
  classes=class_names,
  subset="training",
  batch_size=batch_size, 
  shuffle=True,
  seed=10)

validation_set = train_data_gen.flow_from_directory(train_dataset_path,
  target_size=(224, 224),
  class_mode="categorical",
  classes=class_names,
  subset="validation",
  batch_size=batch_size,
  shuffle=True,
  seed=10)

base_model = VGG16(input_shape=(224,224,3), weights="imagenet", include_top=False)

if finetune == "yes":
  for layer in base_model.layers[:-2]:
    layer.trainable = False
else:
  for layer in base_model.layers:
    layer.trainable = False

top_model = base_model.output
top_model = Flatten(name="flatten")(top_model)
top_model = Dense(4096, activation="relu")(top_model)
top_model = Dense(1072, activation="relu")(top_model)
top_model = Dropout(dropout_value)(top_model)
output_layer = Dense(num_classes, activation="softmax")(top_model)

opt = Adam(lr=0.001)

if (finetune == "yes"):
  opt = Adam(learning_rate=0.0001)

model = Model(inputs=base_model.input, outputs=output_layer)

model.compile(
  loss="categorical_crossentropy",
  optimizer=opt,
  metrics=["accuracy"]
)

checkpoint = ModelCheckpoint(filepath="models/" + modelname, save_best_only=True, verbose=1)

early = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True, mode="min")

num_epochs = 100
steps_epoch = train_set.samples // batch_size
steps_val = validation_set.samples // batch_size

history = model.fit(train_set, batch_size=batch_size, 
epochs=num_epochs, validation_data=validation_set, 
steps_per_epoch=steps_epoch,
validation_steps=steps_val,
verbose=1, 
callbacks=[checkpoint, early])

plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("model accuracy")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train", "val"], loc="upper left")
plt.show()



