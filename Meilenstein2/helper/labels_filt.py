import os
import json
import shutil

def count_empty_json_files(folder_path, image_path, output_path):
    empty_file_count = 0
    os.makedirs(output_path, exist_ok=True)

    # Iterate over all files in the folder
    for filename in os.listdir(folder_path):
        # Check if file has a .json extension
        if filename.endswith('.json'):
            file_path = os.path.join(folder_path, filename)
            image_file_path = os.path.join(image_path, filename.replace('.json', '.png'))
            try:
                # Open and load the JSON file
                with open(file_path, 'r') as f:
                    content = json.load(f)

                # Check if the JSON content is empty or just an empty array
                if content == {} or content == []:
                    shutil.move(file_path, output_path)
                    shutil.move(image_file_path, output_path)
                    empty_file_count += 1

            except json.JSONDecodeError:
                # If the file is not valid JSON, skip it
                print(f"Warning: {filename} is not a valid JSON file.")
                continue
            except Exception as e:
                print(f"Error reading {filename}: {e}")
                continue

    return empty_file_count

# Example usage
folder_path = 'data/json'
image_path = 'data/training/faces'
output_path = 'data/filtered'
count = count_empty_json_files(folder_path, image_path, output_path)
print(f"Number of empty or empty-array JSON files: {count}")
