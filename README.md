# Image_Classification

📌 Introduction

This project implements an image classification system capable of automatically identifying and labeling objects present in images such as plane, car, bird, cat, dog, deer, frog, horse, ship, and truck.
The system is trained on the CIFAR-10 dataset, which consists of small labeled color images, and is later used to classify a real-world external image (deer.png).
The project demonstrates the practical application of deep learning and computer vision for real-world image recognition tasks.

🛠️ Technical Stack

1.Programming Language: Python

2.Deep Learning Framework: TensorFlow / Keras

3.Algorithm: Convolutional Neural Network (CNN)

4.Dataset: CIFAR-10

5.Image Processing: OpenCV

6.Visualization: Matplotlib

7.Model Format: .keras (Saved Keras model)

📊 Step-by-Step Project Workflow
1️⃣ Dataset Loading

a.Loaded the CIFAR-10 dataset containing 60,000 RGB images.

b.Each image has a fixed size of 32 × 32 × 3.

c.The dataset contains 10 object classes.

Purpose:
To provide labeled training data so the model can learn visual patterns for each class.

2️⃣ Data Preprocessing

Normalized image pixel values from 0–255 to 0–1.

This improves training stability and learning speed of the CNN.

3️⃣ Data Visualization

Displayed sample images with their class labels.

Verified correct data loading and labeling.

Helped in understanding image patterns and data distribution.

4️⃣ Model Selection (CNN)

Used a Convolutional Neural Network (CNN) for classification.

CNN automatically learns features such as:

Edges

Shapes

Textures

Object parts

Why CNN?
CNNs preserve spatial relationships between pixels and outperform traditional ML algorithms for image data.

5️⃣ Model Loading

Loaded a pre-trained CNN model saved as Image_Classification.keras.

The model contains:

Convolution layers

ReLU activation

Pooling layers

Fully connected layers

Softmax output layer

6️⃣ External Image Prediction

Loaded an external image (deer.png).

Converted color format (BGR → RGB).

Resized the image to 32 × 32 to match training data.

Normalized pixel values.

Added a batch dimension before prediction.

Passed the image to the model for inference.

7️⃣ Classification Output

The model outputs probabilities for all 10 classes.

The class with the highest probability is selected as the prediction.

The predicted class and confidence score are displayed along with the image.

Example output:

Prediction: Deer (92.45%)

🧪 Learning Approach

Learning Type: Supervised Learning

The model learns by comparing predicted labels with true labels during training and minimizing error using backpropagation.

✅ Conclusion

This project demonstrates a complete end-to-end image classification pipeline using deep learning.
It covers dataset handling, image preprocessing, CNN-based feature extraction, model inference, and real-world image prediction.
The system successfully classifies unseen images, showcasing practical skills in computer vision, deep learning, and model deployment, making it suitable for academic, portfolio, and interview presentation.
