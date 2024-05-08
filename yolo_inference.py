from ultralytics import YOLO

#specify the version and model https://github.com/ultralytics/ultralytics
model = YOLO('models/best.pt')

#return the results into a variable and save them into an array
results = model.predict('input_videos/08fd33_4.mp4', save=True)



print(results[0])
print("====================")
for box in results[0].boxes:
    print(box)