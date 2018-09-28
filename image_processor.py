import os

from imageai.Detection import ObjectDetection
from imageai.Prediction import ImagePrediction

resNetModel = 'models/DenseNet-BC-121-32.h5'
resNetDetectionModel = 'models/resnet50_coco_best_v2.0.1.h5'

execution_path = os.getcwd()
model_path = os.path.join(execution_path, resNetModel)
detection_model_path = os.path.join(execution_path, resNetDetectionModel)

# Predictions
prediction = ImagePrediction()
prediction.setModelTypeAsDenseNet()
prediction.setModelPath(model_path)
prediction.loadModel()

# Detections
detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath(detection_model_path)
detector.loadModel()


def predict_types(img):
    predictions, probabilities = prediction.predictImage(img.strip(), result_count=5)
    return list(zip(predictions, probabilities))


def detect_objects(img):
    return detector.detectObjectsFromImage(input_image=img.strip(),
                                           output_image_path="images/analyzed_image.jpg",
                                           minimum_percentage_probability=30)
