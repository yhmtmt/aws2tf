import tensorflow as tf
from utils import label_map_util
import cv2

class Odet:
    def __init__(self, label_file='mscoco_label_map.pbtxt', graph_file='frozen_inference_graph.pb'):
        self.label_map = label_map_util.load_labelmap(label_file)
        self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=90, use_display_name=True)
        self.category_index=label_map_util.create_category_index(self.categories)
        
        config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))

        with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
            graph_def=tf.GraphDef()
            graph_def.ParseFromString(f.read())
        
        self.sess=tf.Session(config=config)
        self.sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

    def proc(self, frm):
        rows=frm.shape[0]
        cols=frm.shape[1]
        #inp = cv2.resize(frm, (960, 540))
        inp=frm.copy()
        inp=inp[:,:,[2,1,0]]
        out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
                        self.sess.graph.get_tensor_by_name('detection_scores:0'),
                        self.sess.graph.get_tensor_by_name('detection_boxes:0'),
                        self.sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0':inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
                
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            if classId in self.category_index.keys():
                oname=self.category_index[classId]['name']
            else:
                oname='Unknown'
                
            score = float(out[1][0][i])
            bbox=[float(v) for v in out[2][0][i]]
                
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                cv2.rectangle(frm, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frm, oname, (int(x), int(y)+20), font ,1, (0, 255, 0), 2, cv2.LINE_AA)            
        
