from pathlib import Path
import shutil
file_paths = []

agestr = ""

for age in range(0, 72, 4):
    with open(f"data/labels_for_filling_backup/{age}_male.txt", 'r') as f:
        print(f"data/labels_for_filling_backup/{age}_male.txt")
        path = f.readlines()[0]
        file_paths.append(path)
        img_name = Path(path).name.replace('txt', 'png')
        seed = Path(path).stem[4:]
        agestr += f"{seed},"
        # shutil.copy(
        #     f"/home/pmayer1/Deep-Learning-2024/Meilenstein3/generatedImages{img_name}",
        #     f"data/paddy_video"
        # )
        
print(agestr)

with open('morph.txt', 'w') as file:
    for path in file_paths:
        file.write(path) 
