from flask import Flask, flash, request, redirect, url_for, render_template
import urllib.request
from werkzeug.utils import secure_filename
import os
from keras.preprocessing import image
import cv2


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
"""

Object Detection (On Image) From TF2 Saved Model
=====================================

"""

# PROVIDE PATH TO IMAGE DIRECTORY


# PROVIDE PATH TO MODEL DIRECTORY
PATH_TO_MODEL_DIR = 'model'

# PROVIDE PATH TO LABEL MAP
PATH_TO_LABELS = 'data/coffee_labelmap.pbtxt'

# PROVIDE THE MINIMUM CONFIDENCE THRESHOLD
MIN_CONF_THRESH = float(0.60)

# LOAD THE MODEL

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR +'/saved_model'

print('Loading model...', end='')

# set loading as false

loading = False

def load_model():
    start_time = time.time()

    # LOAD SAVED MODEL AND BUILD DETECTION FUNCTION
    localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
    
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))
    return detect_fn

# LOAD LABEL MAP DATA FOR PLOTTING

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

import numpy as np
from PIL import Image
import warnings
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
      path: the file path to the image
    Returns:
      uint8 numpy array with shape (img_height, img_width, 3)
    """
    image = cv2.imread(path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 640))
    
    return image

def return_image(IMG):
    image = cv2.imread(IMG)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_expanded = np.expand_dims(image_rgb, axis=0)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    tf.keras.backend.clear_session()
    loaded_model = load_model()
    detections = loaded_model(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_with_detections = image.copy()

    # SET MIN_SCORE_THRESH BASED ON YOU MINIMUM THRESHOLD FOR DETECTIONS
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'],
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=200,
        min_score_thresh=0.5,
        agnostic_mode=False)

    print('Done')
    
    return image_with_detections,detections


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
PRED_FOLDER = 'static/predictions'
 
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['PRED_FOLDER'] = PRED_FOLDER 
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

@app.route("/home")
@app.route("/")
def hello_world():
    return render_template('home.html')

# helper functions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

     
@app.route('/submit', methods=['POST','GET'])
def upload_image():
    files = request.files
    print(files)
    if 'image' not in files:
        flash('No file part')
        return render_template('error.html')
    file = files['image']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        
        loading = True
        filename = secure_filename(file.filename)
        
        # image paths - pred and raw image
        download_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        pred_path = os.path.join(app.config['PRED_FOLDER'], filename)
        
        file.save(download_path)
        #print('upload_image filename: ' + filename)
        
        # predict the image
        image_with_detections,detections = return_image(download_path)
        
        # Using cv2.imwrite() method
        # Saving the image
        cv2.imwrite(pred_path, image_with_detections)
        
        flash('Image successfully uploaded and displayed below')
        
        loading = False
        
        return render_template('submit.html', filename=filename,download_path=pred_path)
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)
    
if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)