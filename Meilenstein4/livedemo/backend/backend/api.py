from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
import os
from backend.ml import MLManager
from uuid import uuid4
import json
from PIL import Image
import traceback

app = Flask(__name__)
CORS(app)

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Flask API!"})

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

    recognize_param = request.args.get('recognize', 'false').lower()
    save_on_recognize = request.args.get('save_on_recognize', 'false').lower()
    image_path, name = save_image(image, recognize_param)
    embedding = ml_manager.calc_embedding(image_path)

    # Check for the 'recognize' query parameter -> Default to 'false' if not provided
    if recognize_param == 'true':
        closest, furthest = ml_manager.recognize(embedding)
        if save_on_recognize:
            save_embedding(embedding, name)
        return jsonify({"message": "Image uploaded and saved successfully", "closest": closest, "furthest": furthest}), 200
    else:
        save_embedding(embedding, name)
        return jsonify({"message": "Image uploaded and saved successfully!"}), 200


@app.route('/api/recognize_img')
def api_recognize_img():
    img_name = request.args.get('image', '').lower()
    if img_name == '':
        return jsonify({"error": "No image file provided"}), 400
    try:
        img_path = f'images/{img_name}'
        embedding = ml_manager.calc_embedding(img_path)
        closest, furthest = ml_manager.recognize(embedding)
    except Exception as error:
        traceback.print_exc()
        return jsonify({"error": str(error)}), 500
    return jsonify({"message": "Image uploaded and saved successfully", "closest": closest, "furthest": furthest}), 200

@app.route('/api/reference', methods=['GET'])
def api_reference():
    img = None
    for node in os.scandir('images'):
        if node.is_file() and "reference" in node.name:
            img = node.name
    if img is None:
        jsonify({"message": "No reference image found!!"}), 404
    return jsonify({"message": "Found reference image!", "ref_image": img}), 200

def save_image(image, recognize_param):
    suffix = image.filename.split('.')[-1]
    name = str(uuid4()) + "." + suffix
    image_path = os.path.join(UPLOAD_FOLDER, name)
    image.save(image_path)

    # if image is used for recognition, save it as reference for web frontend
    if recognize_param:
        for node in os.scandir("images"):
            if "reference" in node.name and node.is_file():
                os.remove(node.path)

        with Image.open(image_path) as reloaded_image:
            ref_path = os.path.join("images", f"reference_{str(uuid4()) + '.' + suffix}")
            reloaded_image.save(ref_path)

    return image_path, name

def save_embedding(embedding, name):
    ml_manager.embeddings[f"new/{name}"] = embedding.tolist()

    with open('db.json', 'w') as file:
        json.dump(ml_manager.embeddings, file, indent=4)

def allowed_file(filename):
    allowed_extensions = {'png', 'jpg', 'jpeg', 'gif'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

if __name__ == '__main__':
    UPLOAD_FOLDER = './images/new'
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    ml_manager = MLManager()
    app.run(host='0.0.0.0', port=5000)
