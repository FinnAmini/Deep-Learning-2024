import os
import shutil

os.makedirs("data/test", exist_ok=True)
os.makedirs("data/test/faces", exist_ok=True)
os.makedirs("data/test/places", exist_ok=True)


for node in os.scandir("data/training/faces"):
    if int(node.name.split(".")[0]) >= 62523:
        target_path = f"data/test/faces/{node.name}"
        shutil.move(node.path, target_path)
        # print(node.path, target_path)
        # break
    # break


for node in os.scandir("data/training/places"):
    if int(node.name.split(".")[0]) >= 63000:
        target_path = f"data/test/places/{node.name}"
        shutil.move(node.path, target_path)
        # print(node.path, target_path)
        # break
    # break





