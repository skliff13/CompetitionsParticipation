import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.layers import Input, Convolution3D, MaxPooling3D, UpSampling3D, Concatenate, Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import plot_model


def vgg16_3d(input_shape, out_channels=1, cut_layers=0, use_dropout=False) -> Model:
    base = VGG16(include_top=False, weights='imagenet')
    anchors = {'block1_conv2', 'block2_conv2', 'block3_conv3', 'block4_conv3', 'block5_conv3'}
    model = compose_vgg_model(anchors, base, cut_layers, input_shape, out_channels, use_dropout)

    return model


def vgg19_3d(input_shape, out_channels=1, cut_layers=0, use_dropout=False) -> Model:
    base = VGG19(include_top=False, weights='imagenet')
    anchors = {'block1_conv2', 'block2_conv2', 'block3_conv4', 'block4_conv4', 'block5_conv4'}
    model = compose_vgg_model(anchors, base, cut_layers, input_shape, out_channels, use_dropout)

    return model


def compose_vgg_model(anchors, base, cut_layers, input_shape, out_channels, use_dropout):
    anchor_layers = []
    input_layer = Input(shape=input_shape, name='input_3d')
    x = input_layer
    weights_to_copy = {}
    for layer in base.layers[:len(base.layers) - cut_layers]:
        if layer.__class__ == Conv2D:
            ws = layer.get_weights()
            ws[0] = np.sum(ws[0], axis=2, keepdims=True) if ws[0].shape[2] == 3 else ws[0]
            ws[0] = np.expand_dims(ws[0], 2)
            shp = ws[0].shape
            nm = layer.name + '_3d'
            print(f'Creating Conv3D layer {nm} with weights shape {shp}')
            ks = shp[:2] + (1,)
            new_layer = Convolution3D(filters=shp[-1], kernel_size=ks, padding='same', activation='relu', name=nm)
            weights_to_copy[nm] = ws
            x = new_layer(x)
            if layer.name in anchors:
                anchor_layers.append(x)
        elif layer.__class__ == MaxPooling2D:
            nm = layer.name + '_3d'
            print('Creating MaxPooling3D layer ' + nm)
            x = MaxPooling3D((2, 2, 1), name=nm)(x)

    anchor_layers = anchor_layers[:-1]
    for i, _ in enumerate(anchor_layers):
        x = UpSampling3D(size=(2, 2, 1), name=f'up_{i}')(x)
        layer = anchor_layers[-1 - i]
        x = Concatenate(axis=-1)([layer, x])
        filters = layer.shape[-1]
        nm = f'conv_up{i}-1'
        x = Convolution3D(filters=filters, kernel_size=(3, 3, 3), padding='same', activation='relu', name=nm)(x)
        nm = f'conv_up{i}-2'
        x = Convolution3D(filters=filters, kernel_size=(3, 3, 1), padding='same', activation='relu', name=nm)(x)

    if use_dropout:
        x = Dropout(0.5)(x)
    output = Convolution3D(filters=out_channels, kernel_size=1, padding='same', activation='sigmoid',
                           name='predictions')(x)

    model = Model(input_layer, output)
    for layer in model.layers:
        if layer.name in weights_to_copy:
            print('Copying weights to ' + layer.name)
            layer.set_weights(weights_to_copy[layer.name])
            layer.trainable = False

    return model


if __name__ == '__main__':
    model_ = vgg19_3d((192, 192, 64, 1), out_channels=1, cut_layers=11)
    model_.summary()
    try:
        plot_model(model_, 'tmp_model.png', show_shapes=True)
        print('Saved to tmp_model.png')
    except:
        print('Failed to `plot_model`')
