import numpy as np
import cv2
import os
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Constants
IMG_HEIGHT, IMG_WIDTH = 64, 64

# Function to load images from a directory
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith(".png") or img_path.endswith(".jpg"):
            img = load_img(img_path, color_mode='grayscale', target_size=(IMG_HEIGHT, IMG_WIDTH))
            img = img_to_array(img)
            images.append(img)
    return np.array(images)

# Load training and test data
train_images = load_images_from_folder('data\\train')
test_images = load_images_from_folder('data\\test')

# Preprocess the images
train_images = train_images.astype('float32') / 255.
test_images = test_images.astype('float32') / 255.
train_images = np.reshape(train_images, (len(train_images), IMG_HEIGHT, IMG_WIDTH, 1))
test_images = np.reshape(test_images, (len(test_images), IMG_HEIGHT, IMG_WIDTH, 1))

# Advanced Model Design
def build_advanced_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = BatchNormalization()(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_advanced_autoencoder((IMG_HEIGHT, IMG_WIDTH, 1))

# Model Training
autoencoder.fit(train_images, train_images, epochs=50, batch_size=128, shuffle=True)

# Anomaly Detection Function
def compute_reconstruction_error(data, model):
    reconstructions = model.predict(data)
    reconstruction_errors = tf.keras.losses.mse(reconstructions, data)
    mean_errors = np.mean(reconstruction_errors, axis=(1, 2))
    return mean_errors

# Compute reconstruction error for test images
test_errors = compute_reconstruction_error(test_images, autoencoder)

# Determine threshold based on training data
threshold = np.mean(test_errors) + 1 * np.std(test_errors)

# Classify test images based on reconstruction error
def classify_errors(errors, threshold):
    return ["anomaly" if error > threshold else "normal" for error in errors]

classified_test = classify_errors(test_errors, threshold)

# Display Results
print("Threshold: ", threshold)
print("Test Errors: ", test_errors)
print("Classified Test Images: ", classified_test)
