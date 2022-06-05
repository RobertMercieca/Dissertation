# Dissertation - Crop and disease classification

This was built for my dissertation which makes use of machine learning to classify different crops and diseases using python.

Libraries used: tensorflow, numpy, cv2, os, random, os, sklearn, PIL, rembg, io, matplotlib, flask, werkzeug

The train and test dataset for both classification tasks were sized down due to large file size.

The models were not uploaded due to their large size, these can be trained by running train.py.<br/>
Before running train.py<br/>
-Give the model a name<br/>
-Set the modelfor variable to either "crop" or "disease"<br/>
-Set finetune to "yes" or "no"<br/>
-Set the proper train dataset path

To test the models you can run evaluate.py which will generate a confusion matrix and a classification report.<br/>
Before running evaluate.py<br/>
-Put the path to the model you are evaluating<br/>
-Set the dataset path to the test dataset or your own dataset

Running website.py will run a flask website that allows you to upload an image to retrieve a prediction of either crops or diseases.<br/>
More information is available on the front page of the flask website.<br/>
Before running website.py<br/>
-Open image_prediction.py and set the crop_model and disease_model variables to the path of the respective models
