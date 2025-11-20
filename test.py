import json
import os
import glob
from auto_tagging_engine import AutoTagEngine

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_paths = []
exts = ["*.JPG", "*.jpg"]
for ext in exts:
    image_path_pattern = os.path.join(BASE_DIR, "test_images", ext)
    img_paths = glob.glob(image_path_pattern)
    image_paths.extend(img_paths)

results = AutoTagEngine.do_tagging_process(image_paths)

for image, result in results.items():
    print(image + ' is labeled')
    print(result)

outfile_name = 'result.json'
outfile_path = os.path.join(BASE_DIR, outfile_name)

serializable_results = {}
for k, v in results.items():
    key = os.path.basename(k)
    if isinstance(v, (list, tuple)):
        value = v
    elif hasattr(v, 'tolist'):
        value = v.tolist()
    else:
        value = v
    serializable_results[key] = value

with open(outfile_path, 'w') as outfile:
    json.dump(serializable_results, outfile)
