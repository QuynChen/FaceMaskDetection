from mmdet.apis import init_detector, inference_detector
import cv2
from collections import Counter
import time

class Detector:
    def __init__(self, config_path = None, weight_path = None, threshold = 0.5, label_names = None) -> None:
        self.threshold = threshold
        self.config_path = config_path
        self.weight_path = weight_path
        self.label_names = label_names
        self.load_model()

    def load_model(self):
        self.model = init_detector(self.config_path, self.weight_path, device='cpu')
        
    def score_threshold(self, scores):
        list = []
        for e, score in enumerate(scores):
            if score >= self.threshold:
                list.append(e)
        return list

    def draw_bbox(self, img, label, score, bbox):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        if self.label_names is None:
            self.label_names = ['with_mask',
                'without_mask',
                'mask_weared_incorrect']
        
        text = self.label_names[label] + '  ' + str(round(score))

        x1 = bbox[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]

        x1 = round(x1.item())
        y1 = round(y1.item())
        x2 = round(x2.item())
        y2 = round(y2.item())
        
        img = cv2.rectangle(img, (x1, y1), (x2, y2), colors[label], 3)
        img = cv2.putText(img, text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[label], 2)

        return img

    def detect(self, img):
        start = time.time()
        result = inference_detector(self.model, img)
        end = time.time()
        img_cv = cv2.imread(img)

        labels = result.pred_instances.labels
        scores = result.pred_instances.scores
        bboxes = result.pred_instances.bboxes

        chosen_bbox = self.score_threshold(scores)
        
        chosen_labels = []
        chosen_scores = []
        chosen_bboxes = []

        for index in chosen_bbox:
            chosen_labels.append(labels[index].item())
            chosen_scores.append(scores[index].item())
            chosen_bboxes.append(bboxes[index])

        for e, (label, score, bbox) in enumerate(zip(chosen_labels, chosen_scores, chosen_bboxes)):
            img_cv = self.draw_bbox(img_cv, label, score, bbox)
        class_counts = Counter(chosen_labels)
        timee = end - start 
        return img_cv, class_counts, timee