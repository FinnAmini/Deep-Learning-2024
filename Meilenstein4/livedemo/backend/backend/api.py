from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from backend.ml import MLManager
from uuid import uuid4
import json

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './images/new'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ml_manager = MLManager()

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

# @app.route('/images/<path:filename>')
# def serve_image(filename):
#     print(filename)
#     return send_from_directory(f'images', filename)

@app.route('/images/<path:filename>')
def get_dynamic_image(filename):
    print(filename)
    return send_file(f'../images/{filename}', mimetype='image/png')

@app.route('/api/images')
def get_image_paths():
    images_dir = os.path.join('images')
    image_data = {}

    for person_folder in os.listdir(images_dir):
        person_path = os.path.join(images_dir, person_folder)

        if os.path.isdir(person_path):
            images = os.listdir(person_path)
            image_files = [f'images/{person_folder}/{image}' for image in images if
                           os.path.isfile(os.path.join(person_path, image))]
            image_data[person_folder] = image_files

    return jsonify(image_data)

@app.route('/api/recognize', methods=['POST'])
def api_recognize():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    image = request.files['image']
    if image.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(image.filename):
        return jsonify({"error": "Invalid file type"}), 400
    
    file_path = 'db.json'
    if not os.path.exists(file_path):
        with open(file_path, 'w') as file:
            json.dump({}, file)

    recognize_param = request.args.get('recognize', 'false').lower()
    image_path, name = save_image(image)
    embedding = ml_manager.calc_embedding(image_path)

    # Check for the 'recognize' query parameter -> Default to 'false' if not provided
    if recognize_param == 'true':
        closest, furthest = ml_manager.recognize(embedding)
        save_embedding(embedding, name)
        return jsonify({"message": "Image uploaded and saved successfully", "closest": closest, "furthest": furthest}), 200
    else:
        save_embedding(embedding, name)
        return jsonify({"message": "Image uploaded and saved successfully!"}), 200



def save_image(image):
    name = str(uuid4()) + "." + image.filename.split('.')[-1]
    image_path = os.path.join(UPLOAD_FOLDER, name)
    image.save(image_path)
    return image_path, name

def save_embedding(embedding, name):
    with open('db.json', 'r') as file:
        data = json.load(file)

    data[f"new/{name}"] = embedding.tolist()

    with open('db.json', 'w') as file:
        json.dump(data, file, indent=4)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
