import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolo_dataset import YoloDataset
import torch

# 假设你的图像和标签路径如下
img_dir = "C:\\Code\\VSCode_Py\\MachineVision\\manualYolo\\data\\train\\img\\train"
label_dir = "C:\\Code\\VSCode_Py\\MachineVision\\manualYolo\\data\\train\\label\\train"

def yolo_collate_fn(batch):
    imgs, targets = zip(*batch)  # 解包 batch 中的每一对 (img, targets)
    imgs = torch.stack(imgs, dim=0)  # 图像可以堆叠
    return imgs, targets  # 标签保持 list of tensors


# 创建数据集和数据加载器
dataset = YoloDataset(img_dir, label_dir, img_size=640)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0,collate_fn=yolo_collate_fn)



# 从 DataLoader 中取一批数据
for imgs, targets in dataloader:
    for i in range(len(imgs)):
        img = imgs[i].numpy().transpose(1, 2, 0)  # CHW -> HWC
        target = targets[i]

        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for box in target:
            cls, x, y, w, h = box.tolist()
            x1 = (x - w / 2) * 640
            y1 = (y - h / 2) * 640
            box_w = w * 640
            box_h = h * 640

            rect = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f'cls {int(cls)}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))

        plt.title(f"Image {i}")
        plt.show()
    break  # 只测试一批
