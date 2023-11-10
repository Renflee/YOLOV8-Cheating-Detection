from ultralytics import YOLO

model = YOLO('yolov8n.pt')


results = model('images/person.png')  
for result in results:
    boxes = result.boxes  


try:
    print(f'result: {boxes}')
    print(f'boxes: {boxes.data[0]}')
    print(f'class: {boxes.cls[0]}')
    print(f'conf: {boxes.conf[0]}')
except:
    print("Cannot fing class")

# print(f'probs: {probs}')