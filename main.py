import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, models

# -------------------- Load CIFAR-10 Dataset --------------------
(training_img, training_labels), (testing_img, testing_labels) = datasets.cifar10.load_data()

# Normalize images
training_img = training_img / 255.0
testing_img = testing_img / 255.0

# Class names
class_names = ["Plane", "Car", "Bird", "Cat", "Dog",
               "Deer", "Frog", "Horse", "Ship", "Truck"]

# -------------------- Display Sample Training Images --------------------
plt.figure(figsize=(6, 6))
for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_img[i])
    plt.xlabel(class_names[training_labels[i][0]])
plt.tight_layout()
plt.show()

# -------------------- Load Trained Model --------------------
model = models.load_model("Image_Classification.keras")

# Confirm model input shape
print("Model input shape:", model.input_shape)

# -------------------- Load & Predict External Image --------------------
img = cv2.imread("bird.png")

# Convert BGR → RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resize to CIFAR-10 size (32x32)
img = cv2.resize(img, (32, 32))

# Normalize
img = img / 255.0

# Add batch dimension → (1, 32, 32, 3)
img = np.expand_dims(img, axis=0)

# Prediction
prediction = model.predict(img)
index = np.argmax(prediction)
confidence = np.max(prediction) * 100

# -------------------- Display Result --------------------
plt.imshow(img[0])
plt.title(f"Prediction: {class_names[index]} ({confidence:.2f}%)")
plt.axis("off")
plt.show()

print(f"Prediction is: {class_names[index]}")
print(f"Confidence: {confidence:.2f}%")
