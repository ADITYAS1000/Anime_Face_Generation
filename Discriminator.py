from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

class Discriminator():
    def __init__(self, LATENT_DIM, WEIGHT_INIT, CHANNELS):
        self.LATENT_DIM = LATENT_DIM
        self.WEIGHT_INIT = WEIGHT_INIT
        self.CHANNELS = CHANNELS
        self.discriminator = None
    
    def create_model(self):
        model = Sequential(name='discriminator')
        input_shape = (64, 64, 3)
        alpha = 0.2

        # create conv layers
        model.add(layers.Conv2D(64, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=alpha))

        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=alpha))

        model.add(layers.Conv2D(128, (4, 4), strides=(2, 2), padding='same', input_shape=input_shape))
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(alpha=alpha))

        model.add(layers.Flatten())
        model.add(layers.Dropout(0.3))

        # output class
        model.add(layers.Dense(1, activation='sigmoid'))

        self.discriminator = model

        return self.discriminator