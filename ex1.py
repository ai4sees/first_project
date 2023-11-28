import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D

# Synthetic Data Creation
def create_circle_image(diameter, canvas_size=(64, 64)):
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    center = (canvas_size[0] // 2, canvas_size[1] // 2)
    cv2.circle(canvas, center, diameter // 2, (255, 255, 255), -1)
    return canvas

def create_triangle_image(side_length, canvas_size=(64, 64)):
    canvas = np.zeros(canvas_size, dtype=np.uint8)
    pt1 = (canvas_size[0] // 2, canvas_size[1] // 2 - side_length // 2)
    pt2 = (pt1[0] - side_length // 2, pt1[1] + side_length // 2)
    pt3 = (pt1[0] + side_length // 2, pt1[1] + side_length // 2)
    triangle_cnt = np.array([pt1, pt2, pt3])
    cv2.drawContours(canvas, [triangle_cnt], 0, (255, 255, 255), -1)
    return canvas

normal_images = [create_circle_image(20) for _ in range(100)]
anomalous_images = [create_triangle_image(30) for _ in range(10)]

normal_images = np.array(normal_images)
anomalous_images = np.array(anomalous_images)

# Model Design
def build_autoencoder(input_shape):
    input_img = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

autoencoder = build_autoencoder((64, 64, 1))

# Data Preprocessing
normal_images = normal_images.astype('float32') / 255.
anomalous_images = anomalous_images.astype('float32') / 255.
normal_images = np.reshape(normal_images, (len(normal_images), 64, 64, 1))
anomalous_images = np.reshape(anomalous_images, (len(anomalous_images), 64, 64, 1))

# Model Training
autoencoder.fit(normal_images, normal_images, epochs=50, batch_size=128, shuffle=True)

# Anomaly Detection
def compute_reconstruction_error(data, model):
    reconstructions = model.predict(data)
    reconstruction_errors = tf.keras.losses.mse(reconstructions, data)
    return reconstruction_errors.numpy()

normal_errors = compute_reconstruction_error(normal_images, autoencoder)
anomaly_errors = compute_reconstruction_error(anomalous_images, autoencoder)

threshold = np.mean(normal_errors) + 2 * np.std(normal_errors)

def classify_errors(errors, threshold):
    return ["anomaly" if error > threshold else "normal" for error in errors]

classified_normal = classify_errors(normal_errors, threshold)
classified_anomalies = classify_errors(anomaly_errors, threshold)

# Display Results
print("Threshold: ", threshold)
print("Normal Errors: ", normal_errors)
print("Anomaly Errors: ", anomaly_errors)
print("Classified Normal: ", classified_normal)
print("Classified Anomalies: ", classified_anomalies)
