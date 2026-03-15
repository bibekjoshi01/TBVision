import cv2
import numpy as np


class LungPreprocessor:
    def __init__(self, image_size=224):
        self.image_size = image_size

    def remove_borders(self, img):
        h, w = img.shape
        border = int(min(h, w) * 0.03)
        return img[border : h - border, border : w - border]

    def normalize_intensity(self, img):
        p2, p98 = np.percentile(img, (2, 98))
        img = np.clip(img, p2, p98)
        img = ((img - img.min()) / (max(1e-8, img.max() - img.min())) * 255).astype(
            np.uint8
        )
        return img

    def apply_clahe(self, img):
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(img)

    def lung_mask(self, img):
        """simple threshold-based lung mask"""
        blur = cv2.GaussianBlur(img, (5, 5), 0)
        _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = cv2.bitwise_not(mask)  # lungs are dark
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        )
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)),
        )
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        mask_float = mask.astype(float) / 255.0
        return (img * mask_float + img * 0.2 * (1 - mask_float)).astype(np.uint8)

    def preprocess(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Image not found: {img_path}")

        img = self.remove_borders(img)
        img = self.lung_mask(img)
        img = self.normalize_intensity(img)
        img = self.apply_clahe(img)
        img = cv2.resize(img, (self.image_size, self.image_size))
        return img
