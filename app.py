from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import os

app = Flask(__name__)
CORS(app)

model = YOLO("./model.pt")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file found'}), 400

    image = request.files['image']
    image_path = f"temp/{image.filename}"

    os.makedirs("temp", exist_ok=True)
    image.save(image_path)

    results = model.predict(source=image_path, conf=0.35)

    predictions = []
    count = 0

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        confidences = result.boxes.conf.cpu().numpy()  # Confidence scores
        classes = result.boxes.cls.cpu().numpy()  # Class IDs
        for box, confidence, cls in zip(boxes, confidences, classes):
            count += 1
            predictions.append({
                "box": box.tolist(),
                "confidence": float(confidence),
                "class": int(cls)
            })

    # Geçici dosyayı silin
    os.remove(image_path)

    # Sonuçları JSON olarak döndürün
    return jsonify({'predictions': predictions, 'count': count})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
