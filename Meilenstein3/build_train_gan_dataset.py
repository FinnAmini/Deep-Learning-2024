import os
import shutil
from concurrent.futures import ThreadPoolExecutor

# Define your directories
labels_dir = "Meilenstein2/data/labels_for_filling"  # Folder containing .txt files
output_dir = "Meilenstein2/data/labels_gans/train"       # Output folder for copied images
image_base_path = "Meilenstein2/data/generatedLabels"  # Base path for images in .png format

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_line(json_path):
    """
    Processes a single line (json path), converts it to a png path, 
    and copies the file to the output directory.
    """
    json_path = json_path.strip()  # Remove whitespace

    if len(json_path) != 0:
        # png_path = json_path.replace(".json", ".png")
        png_path = json_path

        # Build full paths
        full_png_path = os.path.join(image_base_path, os.path.basename(png_path))
        destination_path = os.path.join(output_dir, os.path.basename(png_path))

        # Copy the image if it exists
        if os.path.exists(full_png_path):
            shutil.copy(full_png_path, destination_path)
            # return f"Copied: {full_png_path} -> {destination_path}"
        else:
            return f"Image not found: {full_png_path}"

def process_file(txt_file_path):
    """
    Processes all lines in a given .txt file.
    """
    results = []
    with open(txt_file_path, "r") as txt_file:
        for line in txt_file:
            results.append(line.strip())
    return results

# Gather all lines from all files
all_tasks = []
for filename in os.listdir(labels_dir):
    if filename.endswith(".txt"):
        txt_file_path = os.path.join(labels_dir, filename)
        all_tasks.extend(process_file(txt_file_path))

# Use ThreadPoolExecutor for multi-threading
with ThreadPoolExecutor() as executor:
    results = list(executor.map(process_line, all_tasks))

# Output the results
for result in results:
    print(result)