from ultralytics import YOLO

# Load the pretrained YOLO11 nano model
model = YOLO("yolo11n.pt") 
def main():
# Train the model
    results = model.train(
        data="C:/NMS/comp-vision/boxing/BoxingHub.v3i.yolo26/data.yaml", 
        epochs=50, 
        imgsz=640, 
        device=0,
    )

if __name__ == '__main__':
    main()