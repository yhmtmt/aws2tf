import numpy as np
import tensorflow as tf
import cv2 as cv

from utils import label_map_util

label_map = label_map_util.load_labelmap('mscoco_label_map.pbtxt')
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=9, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

# Read the graph.
with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session(config=config) as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread('image.jpg')
    img = cv.resize(img, (1024,768))
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (400, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
    num_detections = int(out[0][0])
    for i in range(num_detections):
        classId = int(out[3][0][i])
        if classId in category_index.keys():
            oname=category_index[classId]['name']
            print("%d th object is %s" % (i, category_index[classId]['name']))
        else:
            oname="None"
            print("%dth object, no such key %d" % (i, classId))
            
        score = float(out[1][0][i])
        bbox = [float(v) for v in out[2][0][i]]
        if score > 0.3:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            font = cv.FONT_HERSHEY_SIMPLEX
            cv.putText(img, oname,  (int(x), int(y)), font, 1, (0,255,0), 2, cv.LINE_AA)

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()
