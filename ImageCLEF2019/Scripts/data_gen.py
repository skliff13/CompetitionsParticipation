from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator


class ModifiedDataGenerator(ImageDataGenerator):
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 zca_epsilon=1e-6,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 preprocessing_function=None,
                 data_format=None,
                 crop_to=-1):

        ImageDataGenerator.__init__(self, featurewise_center=featurewise_center,
                                    samplewise_center=samplewise_center,
                                    featurewise_std_normalization=featurewise_std_normalization,
                                    samplewise_std_normalization=samplewise_std_normalization,
                                    zca_whitening=zca_whitening,
                                    zca_epsilon=zca_epsilon,
                                    rotation_range=rotation_range,
                                    width_shift_range=width_shift_range,
                                    height_shift_range=height_shift_range,
                                    shear_range=shear_range,
                                    zoom_range=zoom_range,
                                    channel_shift_range=channel_shift_range,
                                    fill_mode=fill_mode,
                                    cval=cval,
                                    horizontal_flip=horizontal_flip,
                                    vertical_flip=vertical_flip,
                                    rescale=rescale,
                                    preprocessing_function=preprocessing_function,
                                    data_format=data_format)

        self.crop_to = crop_to

    def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='png'):
        return ModifiedNumpyArrayIterator(
            x, y, self,
            batch_size=batch_size,
            shuffle=shuffle,
            seed=seed,
            data_format=self.data_format,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format=save_format,
            crop_to=self.crop_to)

    def crop_data_bacth(self, x):
        sz = self.crop_to
        off1 = (x.shape[1] - sz) // 2
        off2 = (x.shape[2] - sz) // 2

        if off1 >= 0 and off2 >= 0:
            x = x[:, off1:off1 + sz, off2:off2 + sz, :]

        return x


class ModifiedNumpyArrayIterator(NumpyArrayIterator):
    def __init__(self, x, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 data_format=None,
                 save_to_dir=None, save_prefix='', save_format='png', crop_to=-1):

        NumpyArrayIterator.__init__(self, x, y, image_data_generator,
                                    batch_size=batch_size, shuffle=shuffle, seed=seed,
                                    data_format=data_format,
                                    save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)
        self.crop_to = crop_to

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x, batch_y = super()._get_batches_of_transformed_samples(index_array)

        if self.crop_to > 0:
            batch_x = self.image_data_generator.crop_data_bacth(batch_x)

        return batch_x, batch_y


if __name__ == '__main__':
    from train_model import load_data

    data_dir = '/home/skliff13/work/PTD_Xray/datasets/tuberculosis/v2.2'
    data_shape = (256, 256)
    (x_train, y_train), (x_val, y_val) = load_data(data_dir, data_shape)

    train_gen = ModifiedDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, rescale=1.,
                                      zoom_range=0.2, fill_mode='nearest', cval=0, crop_to=224)
    for q, v in train_gen.flow(x_train, y_train, batch_size=8):
        print(q.shape)
        exit(14)
