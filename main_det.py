import os
import glob
import argparse
import urllib.request
import cv2
import numpy as np


def ensure(p, url):
    if os.path.isfile(p):
        return p
    d = os.path.dirname(p)
    if d and not os.path.isdir(d):
        os.makedirs(d, exist_ok=True)
    urllib.request.urlretrieve(url, p)
    return p


def load_model(base_dir):
    cfg = os.path.join(base_dir, 'yolov3-tiny.cfg')
    weights = os.path.join(base_dir, 'yolov3-tiny.weights')
    names = os.path.join(base_dir, 'coco.names')
    ensure(cfg, 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg')
    ensure(weights, 'https://pjreddie.com/media/files/yolov3-tiny.weights')
    ensure(names, 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names')
    with open(names, 'r') as f:
        classes = [l.strip() for l in f.readlines() if l.strip()]
    net = cv2.dnn.readNetFromDarknet(cfg, weights)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    ln = net.getLayerNames()
    idxs = net.getUnconnectedOutLayers()
    try:
        idxs = idxs.flatten().tolist()
    except Exception:
        idxs = [int(x) for x in idxs]
    out_layers = [ln[i - 1] for i in idxs]
    return net, out_layers, classes


def detect(net, out_layers, img, conf_thres, nms_thres):
    (H, W) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(out_layers)
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = int(np.argmax(scores))
            confidence = float(scores[classID])
            if confidence >= conf_thres:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(confidence)
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)
    return boxes, confidences, classIDs, idxs


def draw(img, boxes, confidences, classIDs, idxs, classes):
    if len(idxs) == 0:
        return img
    for i in idxs.flatten():
        (x, y, w, h) = boxes[i]
        c = (0, 255, 0)
        cv2.rectangle(img, (x, y), (x + w, y + h), c, 2)
        label = classes[classIDs[i]] if classIDs[i] < len(classes) else str(classIDs[i])
        text = f"{label} {confidences[i]:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y - 10 if y - 10 > 10 else y + th + 10
        cv2.rectangle(img, (x, ty - th - 6), (x + tw + 6, ty + 4), c, -1)
        cv2.putText(img, text, (x + 3, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return img


def run(input_dir, output_dir, conf_thres, nms_thres):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    net, out_layers, classes = load_model(base_dir)
    os.makedirs(output_dir, exist_ok=True)
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    paths = []
    for e in exts:
        paths.extend(glob.glob(os.path.join(input_dir, e)))
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        boxes, confidences, classIDs, idxs = detect(net, out_layers, img, conf_thres, nms_thres)
        vis = draw(img.copy(), boxes, confidences, classIDs, idxs, classes)
        out = os.path.join(output_dir, os.path.basename(p))
        cv2.imwrite(out, vis)


def main():
    parser = argparse.ArgumentParser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--input_dir', default=os.path.join(base_dir, 'test_images'))
    parser.add_argument('--output_dir', default=os.path.join(base_dir, 'visualized_boxes'))
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.4)
    args = parser.parse_args()
    run(args.input_dir, args.output_dir, args.conf, args.nms)


if __name__ == '__main__':
    main()