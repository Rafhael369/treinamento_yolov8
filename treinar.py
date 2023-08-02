from ultralytics import YOLO

# Load the model.
model = YOLO('yolov8n.pt')

# Training.
model.train(
    data='config.yaml',
    imgsz=416,
    epochs=50,
    batch=16,
    project='treinamento_yolov8',
    name='treino',
)
