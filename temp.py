{"username":"mahaboobandbasha","key":"f9a1a6c863f923aab4559f695aabeafb"}
https://drive.google.com/file/d/1qsGJvEEpH0dYU6u1KkIADP5si16upneF/view?usp=sharing

# Ultralytics YOLO 🚀, AGPL-3.0 license
# DOTA 2.0 dataset https://captain-whu.github.io/DOTA/index.html for object detection in aerial images by Wuhan University
# Example usage: yolo train model=yolov8n-obb.pt data=DOTAv2.yaml
# parent
# ├── ultralytics
# └── datasets
#     └── dota2  ← downloads here (2GB)

# Train/val/test sets as 1) dir: path/to/imgs, 2) file: path/to/imgs.txt, or 3) list: [path/to/imgs1, path/to/imgs2, ..]
path: /content/drive/MyDrive/ML_classes/project_folder/YOLO_DOTA/DOTA_large_500  # dataset root dir
train: /content/drive/MyDrive/ML_classes/project_folder/YOLO_DOTA/DOTA_large_500/images/test # train images (relative to 'path') 1411 images
val: /content/drive/MyDrive/ML_classes/project_folder/YOLO_DOTA/DOTA_large_500/images/val  # val images (relative to 'path') 458 images
test: /content/drive/MyDrive/ML_classes/project_folder/YOLO_DOTA/DOTA_large_500/images/test # test images (optional) 937 images

# Classes for DOTA 2.0
names:
  0: plane
  1: ship
  2: storage tank
  3: baseball diamond
  4: tennis court
  5: basketball court
  6: ground track field
  7: harbor
  8: bridge
  9: large vehicle
  10: small vehicle
  11: helicopter
  12: roundabout
  13: soccer ball field
  14: swimming pool
  15: container crane
  16: airport
  17: helipad
# Download script/URL (optional)
#download: https://github.com/ultralytics/yolov5/releases/download/v1.0/DOTAv2.zip

