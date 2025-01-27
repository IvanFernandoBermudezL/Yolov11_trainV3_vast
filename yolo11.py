from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8 model
model = YOLO("yolo11n.pt")

def main():
    # Train 
    model.train(
    data="", 
    epochs=150, 
    batch=8, 
    imgsz=640, 
    save_period=5)

if __name__ == '__main__':
    main()