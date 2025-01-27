from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8 model
model = YOLO("yolo11n.pt")

def main():
    # Train 
    model.train(
    data="/Users/ivanbermudez/Downloads/Yolov11_prueba_externa/Ditch_Yolov11/data.yaml", 
    epochs=10, 
    batch=8, 
    imgsz=640, 

    device="mps",
    save_period=5)

if __name__ == '__main__':
    main()

