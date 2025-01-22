import os
import threading
from PIL import Image
from queue import Queue

# Function to check the size of an image
def check_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            if img.size != (224, 224):
                print(f"Invalid size: {image_path} ({img.size})")
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Worker function for threads
def worker(queue):
    while not queue.empty():
        image_path = queue.get()
        if image_path.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
            check_image_size(image_path)
        queue.task_done()

# Main function
def main(folder_path, num_threads=4):
    # Create a queue for file paths
    queue = Queue()

    # Walk through the folder and add image paths to the queue
    for root, _, files in os.walk(folder_path):
        for file in files:
            queue.put(os.path.join(root, file))

    # Create and start threads
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=worker, args=(queue,))
        thread.start()
        threads.append(thread)

    # Wait for all threads to finish
    queue.join()
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    folder = "data/vgg_faces2"
    main(folder)
