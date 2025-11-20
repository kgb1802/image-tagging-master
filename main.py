import os
import csv
import argparse
import json
import glob
import cv2
import main_det


def load_results(csv_path):
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        rows = list(reader)
    header = rows[0]
    labels = header[1:]
    data = {}
    for r in rows[1:]:
        name = r[0]
        scores = [float(x) for x in r[1:]]
        data[name] = scores
    return labels, data


def find_image_path(input_dir, filename):
    p = os.path.join(input_dir, filename)
    if os.path.isfile(p):
        return p
    names = {n.lower(): n for n in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, n))}
    k = filename.lower()
    if k in names:
        return os.path.join(input_dir, names[k])
    return None


def color(i):
    palette = [
        (255, 99, 71),
        (60, 179, 113),
        (65, 105, 225),
        (238, 130, 238),
        (255, 165, 0),
        (70, 130, 180),
        (154, 205, 50),
        (0, 191, 255),
        (255, 215, 0),
        (199, 21, 133),
    ]
    return palette[i % len(palette)]


def draw_tags(img, tags):
    h, w = img.shape[:2]
    x = 10
    y = 30
    for i, (label, score) in enumerate(tags):
        text = f"{label} {score:.3f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x - 5, y - th - 5), (x + tw + 5, y + 5), color(i), -1)
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
        y += th + 14
        if y > h - 10:
            y = 30
            x = min(x + tw + 30, w - 10)
    return img


def top_k_tags(labels, scores, k, threshold):
    pairs = [(labels[i], scores[i]) for i in range(len(labels))]
    pairs.sort(key=lambda x: x[1], reverse=True)
    out = []
    for label, s in pairs:
        if len(out) >= k:
            break
        if s >= threshold:
            out.append((label, s))
    if not out:
        out = pairs[:k]
    return out


def list_images(input_dir):
    exts = ['*.jpg', '*.JPG', '*.jpeg', '*.JPEG', '*.png', '*.PNG']
    names = []
    for e in exts:
        for p in glob.glob(os.path.join(input_dir, e)):
            names.append(os.path.basename(p))
    return names


def process(csv_path, input_dir, output_dir, k, threshold, show, with_boxes, conf, nms):
    labels, data = load_results(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    net = None
    out_layers = None
    classes = None
    if with_boxes:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        net, out_layers, classes = main_det.load_model(base_dir)
    analysis = []
    all_names = set(data.keys())
    for n in list_images(input_dir):
        all_names.add(n)
    for name in sorted(all_names):
        scores = data.get(name)
        ipath = find_image_path(input_dir, name)
        if not ipath:
            continue
        img = cv2.imread(ipath)
        if img is None:
            continue
        tags = []
        if scores is not None:
            tags = top_k_tags(labels, scores, k, threshold)
        vis = img.copy()
        dets = []
        if with_boxes and net is not None:
            boxes, confidences, classIDs, idxs = main_det.detect(net, out_layers, vis, conf, nms)
            if len(idxs) != 0:
                for i in idxs.flatten():
                    (x, y, w, h) = boxes[i]
                    lab = classes[classIDs[i]] if classIDs[i] < len(classes) else str(classIDs[i])
                    dets.append({"label": lab, "confidence": float(confidences[i]), "bbox": [int(x), int(y), int(w), int(h)]})
            vis = main_det.draw(vis, boxes, confidences, classIDs, idxs, classes)
        vis = draw_tags(vis, tags)
        out_path = os.path.join(output_dir, name)
        cv2.imwrite(out_path, vis)
        analysis.append({"image": name, "tags": [{"label": t[0], "score": float(t[1])} for t in tags], "detections": dets})
        if show:
            cv2.imshow('tags', vis)
            key = cv2.waitKey(0)
            if key == 27:
                break
    if show:
        cv2.destroyAllWindows()
    with open(os.path.join(output_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_tag_result.csv'))
    parser.add_argument('--input_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_images'))
    parser.add_argument('--output_dir', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualized_combined'))
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--threshold', type=float, default=0.1)
    parser.add_argument('--show', action='store_true')
    parser.add_argument('--with_boxes', action='store_true')
    parser.add_argument('--conf', type=float, default=0.3)
    parser.add_argument('--nms', type=float, default=0.4)
    args = parser.parse_args()
    process(args.csv_path, args.input_dir, args.output_dir, args.k, args.threshold, args.show, args.with_boxes, args.conf, args.nms)


if __name__ == '__main__':
    main()