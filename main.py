### main.py
import matplotlib.pyplot as plt
import numpy as np
from utils import preprocess_image
from unet import build_unet
import cv2
import tensorflow as tf
import os
from utils import load_image_and_mask
from tensorflow.keras.models import load_model


# Directories
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, "BSR_bsds500", "BSR", "BSDS500", "data", "images", "train")
mask_dir = os.path.join(script_dir, "BSR_bsds500", "BSR", "BSDS500", "data", "groundTruth", "train")




# Specific images to use
image_ids = ["2092", "8049", "8143", "12003", "12074"]

X, Y = [], []

for img_id in image_ids:
    image_path = os.path.join(image_dir, f"{img_id}.jpg")
    mask_path = os.path.join(mask_dir, f"{img_id}.mat")

    print("Image path:", image_path)
    print("Mask path:", mask_path)

    try:
        img, mask = load_image_and_mask(image_path, mask_path)
        X.append(img)
        Y.append(mask)
    except Exception as e:
        print(f"Error loading {img_id}: {e}")

X = np.array(X)
Y = np.array(Y)

if len(X) == 0:
    print("No images loaded — check your file paths.")
    exit()

# Train U-Net
model = build_unet()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, Y, epochs=10, batch_size=2)

# Save model
model.save("unet_bsds500.h5")

# Load and preprocess image
img = preprocess_image(os.path.join(script_dir, "BSR_bsds500", "BSR", "BSDS500", "data", "images", "train", "2092.jpg"))
img_batch = np.expand_dims(img, axis=0)

# Run traditional methods
edges_canny = cv2.Canny((img.squeeze() * 255).astype(np.uint8), 100, 200)
sobelx = cv2.Sobel(img.squeeze(), cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(img.squeeze(), cv2.CV_64F, 0, 1)
edges_sobel = cv2.magnitude(sobelx, sobely)



model = load_model("unet_bsds500.h5")
pred = model.predict(img_batch)[0, ..., 0]



# Show all
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1)
plt.imshow(img.squeeze(), cmap='gray')
plt.title('Original')

plt.subplot(1, 4, 2)
plt.imshow(edges_canny, cmap='gray')
plt.title('Canny')

plt.subplot(1, 4, 3)
plt.imshow(edges_sobel, cmap='gray')
plt.title('Sobel')


plt.subplot(1, 4, 4)
plt.imshow(pred, cmap='gray')
plt.title('U-Net Prediction')


plt.tight_layout()
plt.show()
