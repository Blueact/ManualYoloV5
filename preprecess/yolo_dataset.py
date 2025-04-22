import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np
import torchvision.transforms as T

class YoloDataset(Dataset):
    def __init__(self, img_dir, label_dir, img_size=640):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_size = img_size
        self.img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]

    def __len__(self):
        return len(self.img_files)

    def letterbox(self, img, new_size=640, color=(114, 114, 114)):
        shape = img.shape[:2]  # (h, w)
        ratio = min(new_size / shape[0], new_size / shape[1])
        new_unpad = (int(round(shape[1] * ratio)), int(round(shape[0] * ratio)))

        img_resized = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        dw = new_size - new_unpad[0]
        dh = new_size - new_unpad[1]
        top, bottom = dh // 2, dh - (dh // 2)
        left, right = dw // 2, dw - (dw // 2)

        img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img_padded, ratio, (left, top)

    def __getitem__(self, index):
        img_filename = self.img_files[index]
        img_path = os.path.join(self.img_dir, img_filename)
        label_path = os.path.join(self.label_dir, img_filename.replace('.jpg', '.txt').replace('.png', '.txt'))

        # 加载图像为 BGR，用 OpenCV 是为了 resize 填充方便
        img_bgr = cv2.imread(img_path)
        assert img_bgr is not None, f"Image not found: {img_path}"

        img_bgr, ratio, (pad_x, pad_y) = self.letterbox(img_bgr, self.img_size)

        # 归一化像素值并转为 CHW 格式
        img = img_bgr[:, :, ::-1]  # BGR -> RGB
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = torch.from_numpy(img)

        # 加载标签
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())

                    # 原始是归一化坐标，要乘以原图大小
                    x *= ratio * img_bgr.shape[1] / self.img_size
                    y *= ratio * img_bgr.shape[0] / self.img_size
                    w *= ratio * img_bgr.shape[1] / self.img_size
                    h *= ratio * img_bgr.shape[0] / self.img_size

                    # 加上 padding 偏移（再除以最终图像尺寸进行归一化）
                    x = (x * self.img_size + pad_x) / self.img_size
                    y = (y * self.img_size + pad_y) / self.img_size
                    w *= ratio
                    h *= ratio

                    targets.append([cls, x, y, w, h])

        targets = torch.tensor(targets, dtype=torch.float32)

        return img, targets
