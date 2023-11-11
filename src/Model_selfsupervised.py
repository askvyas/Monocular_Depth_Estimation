import tensorflow as tf
from tensorflow.keras import layers

class SelfSupervisedModel(tf.keras.Model):
    def __init__(self):
        super(SelfSupervisedModel, self).__init__()
        # Encoder: You can adjust the number and size of layers
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.MaxPooling2D((2, 2), padding='same')
        ])
        
        # Decoder: Mirrors the encoder
        self.decoder = tf.keras.Sequential([
            layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same'),
            layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')  # Assuming grayscale output
        ])
    
    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Now use this model in your existing code
model = SelfSupervisedModel()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
# model.fit(dataset, epochs=10)
