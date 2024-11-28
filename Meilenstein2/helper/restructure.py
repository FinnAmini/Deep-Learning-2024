import os
import shutil

os.makedirs("data/st_data/faces", exist_ok=True)

for node in os.scandir("../ffhq-dataset/images224x224"):
    for img in os.scandir(node.path):
        target_path = f"data/st_data/faces/{img.name}"
        # print(img.path, target_path)
        shutil.copyfile(img.path, target_path)
        # break
    # break
    