from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from backend.ml import MLManager

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = '../frontend/public'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
ml_manager = MLManager()

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400


    image_path = save_image(image)

 # Check for the 'recognize' query parameter
    recognize_param = request.args.get('recognize', 'false').lower()  # Default to 'false' if not provided
    if recognize_param == 'true':
        # Perform recognition
        recognized = ml_manager.recognize(image_path)
        return jsonify({"message": "Image uploaded and recognized successfully", 
                        "path": image_path, 
                        "recognition": recognized}), 200
    else:
        # Only save the image
        return jsonify({"message": "Image uploaded and saved successfully", 
                        "path": image_path}), 200

def save_image(image):
    image_path = os.path.join(UPLOAD_FOLDER, f"reference.{image.filename.split('.')[-1]}")
    image.save(image_path)
    return image_path

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(port=5000)
