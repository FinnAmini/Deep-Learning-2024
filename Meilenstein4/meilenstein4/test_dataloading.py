from dataloading import create_dataloader
import cv2

train_dataloader, val_dataloader = create_dataloader('data/vgg_faces2_resized/train', batch_size=32)

count = 0
for (img1, img2), label in train_dataloader:
    cv2.imwrite(f"data/eval/img1_{count}.jpg", img1[0].numpy())
    cv2.imwrite(f"data/eval/img2_{count}.jpg", img2[0].numpy())
    print(label)
    count += 1
    if count == 5:
        break