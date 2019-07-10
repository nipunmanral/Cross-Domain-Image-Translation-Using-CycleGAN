import numpy as np
from keras.layers import Conv2D, Conv2DTranspose, LeakyReLU, Activation, ZeroPadding2D, BatchNormalization, Add
from keras import Input, Model
from keras_contrib.layers import InstanceNormalization

#Define residual block with 2 convolutional layers and a skip connection
def resnet_block(x):    
    x2 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x)
    x2 = InstanceNormalization(axis=1)(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv2D(filters=256, kernel_size=3, strides=1, padding="same")(x2)
    x2 = InstanceNormalization(axis=1)(x2)

    return Add()([x2, x])

#Define generator network with encoder-transformation-decoder style architecture
def define_generator_network(num_resnet_blocks=9):
    input_size = (128,128,3)
    
    #Input RGB image 
    input_layer = Input(shape=input_size)

    #Down-sampling using conv2d
    x = Conv2D(filters=64, kernel_size=7, strides=1, padding="same")(input_layer)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=128, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=256, kernel_size=3, strides=2, padding="same")(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    #Transforming the hidden representation using the resnet blocks
    for i in range(num_resnet_blocks):
        x = resnet_block(x)
    
    #Upsampling to recover the transformed image
    #Conv2DTranspose with a stride 2 works like Conv2D with stride 1/2
    x = Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False)(x)
    x = InstanceNormalization(axis=1)(x)
    x = Activation("relu")(x)

    x = Conv2D(filters=3, kernel_size=7, strides=1, padding="same")(x)
    output = Activation('tanh')(x) #tanh activation to get normalised output image

    model = Model(inputs=[input_layer], outputs=[output])
    return model

#Define the discriminator network based on the PatchGAN's architecture
def define_discriminator_network():
    input_size = (128, 128, 3)
    num_hidden_layers = 3
    input_layer = Input(shape=input_size)

    x = ZeroPadding2D(padding=(1, 1))(input_layer)

    x = Conv2D(filters=64, kernel_size=4, strides=2, padding="valid")(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = ZeroPadding2D(padding=(1, 1))(x)

    for i in range(1, num_hidden_layers + 1):
        x = Conv2D(filters=2 ** i * 64, kernel_size=4, strides=2, padding="valid")(x)
        x = InstanceNormalization(axis=1)(x)
        x = LeakyReLU(alpha=0.2)(x)

        x = ZeroPadding2D(padding=(1, 1))(x)

    #Sigmoid activation to normalise output values between 0 and 1 which will be used to train real or fake labels
    output = Conv2D(filters=1, kernel_size=4, strides=1, activation="sigmoid")(x) #This is the patch output

    model = Model(inputs=[input_layer], outputs=[output])
    return model

def define_adversarial_model(generatorA2B, generatorB2A, discriminatorA, discriminatorB, train_optimizer, lambda_cyc = 10, lambda_idt = 5):
        
    inA = Input(shape=(128, 128, 3))
    inB = Input(shape=(128, 128, 3))

    fakeB = generatorA2B(inA)
    fakeA = generatorB2A(inB)

    reconstructedA = generatorB2A(fakeB)
    reconstructedB = generatorA2B(fakeA)

    identityA = generatorB2A(inA)
    identityB = generatorA2B(inB)

    decisionA = discriminatorA(fakeA)
    decisionB = discriminatorB(fakeB)

    adversarial_model = Model(inputs = [inA, inB], outputs = [decisionA, decisionB, reconstructedA, reconstructedB, identityA, identityB])
    adversarial_model.compile(loss= ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'], loss_weights= [1, 1, lambda_cyc, lambda_cyc, lambda_idt, lambda_idt],
                                optimizer = train_optimizer)
    print(adversarial_model.summary())
    return adversarial_model




