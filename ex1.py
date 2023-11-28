import tensorflow as tf
from tensorflow.keras import layers

# Define the VAE model
class VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.InputLayer(input_shape=(28, 28, 1)),
            layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(latent_dim * 2),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(latent_dim * 2),
            layers.Reshape((7, 7, 2)),
            layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), activation='relu'),
            layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), activation='relu'),
            layers.Conv2DTranspose(filters=1, kernel_size=(3, 3), activation='sigmoid'),
        ])

    def encode(self, x):
        mean, log_var = tf.split(self.encoder(x), num_or_size_splits=2, axis=-1)
        return mean, log_var

    def decode(self, z):
        return self.decoder(z)

    def reconstruct(self, x):
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        return self.decode(z)

    def reparameterize(self, mean, log_var):
        eps = tf.random.normal(shape=mean.shape)
        return mean + eps * tf.exp(0.5 * log_var)

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Select a small subset of anomalous images
anomalous_images = x_train[y_train == 1][:100]

# Train the VAE model
vae = VAE(latent_dim=32)
vae.compile(optimizer='adam', loss='binary_crossentropy')
vae.fit(x_train, x_train, epochs=10)

# Define anomaly threshold
threshold = 20

# Detect anomalies in test data
reconstructed_images = vae.reconstruct(x_test)
anomaly_scores = tf.keras.backend.mean(tf.keras.backend.square(x_test - reconstructed_images), axis=(1, 2, 3))
anomalous_predictions = tf.cast(anomaly_scores > threshold, tf.float32)

# Evaluate model performance
precision = tf.keras.metrics.precision(y_test, anomalous_predictions)
recall = tf.keras.metrics.recall(y_test, anomalous_predictions)
f1_score = tf.keras.metrics.f1_score(y_test, anomalous_predictions)

print('Precision:', precision.result().numpy())
print('Recall:', recall.result().numpy())
print('F1-score:', f1_score.result().numpy())


