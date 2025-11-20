from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import json

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import *

print (tf.__version__) # Must be v1.1+

model = ResNet50(weights='imagenet')

img = image.load_img('test_images/bike.JPG', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
top5 = decode_predictions(preds, top=5)[0]
print('Resnet Predicted:', top5)

base_dir = os.path.dirname(os.path.abspath(__file__))
outfile_path = os.path.join(base_dir, 'resnet_result.json')
out = [{'id': i, 'name': n, 'score': float(s)} for (i, n, s) in top5]
with open(outfile_path, 'w') as f:
    json.dump(out, f)
