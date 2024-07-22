from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from kmeans import KMeans

def image_seg_clustering(img_path, k=3, max_iters=100):
    img = Image.open(img_path)
    img = np.array(img)
    pixel_values = img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    kmean_seg = KMeans(k=k, max_iters=max_iters)
    y_pred = kmean_seg.predict(pixel_values)
    centers = np.uint8(kmean_seg.cent())
    labels = y_pred.flatten()
    
    segmented_image = centers[labels]
    segmented_image = segmented_image.reshape(img.shape)
    
    return img, segmented_image

def save_images(original, segmented, original_path, segmented_path):
    original_image = Image.fromarray(original)
    segmented_image = Image.fromarray(segmented)
    original_image.save(original_path)
    segmented_image.save(segmented_path)