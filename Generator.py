from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers

class Generator():
    def __init__(self, LATENT_DIM, WEIGHT_INIT, CHANNELS):
        self.LATENT_DIM = LATENT_DIM
        self.WEIGHT_INIT = WEIGHT_INIT
        self.CHANNELS = CHANNELS
        self.generator = None


    def create_model(self):
        
        model = Sequential(name='generator')

        # 1d random noise
        model.add(layers.Dense(8 * 8 * 512, input_dim=self.LATENT_DIM))
        # model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        # convert 1d to 3d
        model.add(layers.Reshape((8, 8, 512)))

        # upsample to 16x16
        model.add(layers.Conv2DTranspose(256, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.WEIGHT_INIT))
        # model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        # upsample to 32x32
        model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.WEIGHT_INIT))
        # model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        # upsample to 64x64
        model.add(layers.Conv2DTranspose(64, (4, 4), strides=(2, 2), padding='same', kernel_initializer=self.WEIGHT_INIT))
        # model.add(layers.BatchNormalization())
        model.add(layers.ReLU())

        model.add(layers.Conv2D(self.CHANNELS, (4, 4), padding='same', activation='tanh'))

        self.generator = model

        return self.generator