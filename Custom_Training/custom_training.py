import os
from ultralytics import YOLO
import yaml
from roboflow import Roboflow

# Downloading the data
rf = Roboflow(api_key="your-api-key")
project = rf.workspace("roboflow-universe-projects").project("license-plate-recognition-rxg4e")
dataset = project.version(4).download("yolov8")

os.chdir(os.path.join(os.getcwd(),'License-Plate-Recognition-4'))


train_path=os.path.join(os.getcwd(),'train','images')
valid_path=os.path.join(os.getcwd(),'valid','images')

file_path = 'data.yaml'
with open(file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

yaml_data['train'] = train_path
yaml_data['val'] = valid_path

with open(file_path, 'w') as file:
    yaml.dump(yaml_data, file, default_flow_style=False)

file_path = 'data.yaml'

with open(file_path, 'r') as file:
    yaml_data = yaml.safe_load(file)

print(yaml_data)

model = YOLO("yolov8n.yaml")

# Use the model
model.train(data="data.yaml", epochs=20)







# import os
#
# from ultralytics import YOLO
# import cv2
#
#
# # VIDEOS_DIR = os.path.join()
#
# video_path = os.path.join(r'C:\Users\Dell\Desktop\Python\Projects\Liscence_Blur\sample.mp4')
# video_path_out = '{}_out.mp4'.format(video_path)
#
# cap = cv2.VideoCapture(video_path)
# ret, frame = cap.read()
# H, W, _ = frame.shape
# out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
#
# model_path = os.path.join('.', 'runs', 'detect', 'train', 'weights', 'last.pt')
#
# # Load a model
# model = YOLO(r'C:\Users\Dell\Desktop\Python\Projects\Liscence_Blur\Custom_Training\best.pt')  # load a custom model
#
# threshold = 0.5
#
# while ret:
#
#     results = model(frame)[0]
#
#     for result in results.boxes.data.tolist():
#         x1, y1, x2, y2, score, class_id = result
#
#         if score > threshold:
#             cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
#             cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)
#
#     out.write(frame)
#     ret, frame = cap.read()
#
# cap.release()
# out.release()
# cv2.destroyAllWindows()