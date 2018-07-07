import numpy as np
import os, sys, random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import keras
from keras import backend as K
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import get_source_inputs
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras import optimizers


def recover_3darrays(emotions_dir, neutral_instances):
    """Generates a single X, y arrays using all Numpy binary file.

      Args:
        emotions_dir: String path to folder with emotion folders.
        neutral_instances: Number of neutral instances to add to the X, y 
            arrays. Given the high number of neutral instances that are
            generated, even with a low class weight in the training phase 
            the model will have a poor performance. A good choice can be 
            a number between 30 - 50.

      Returns:
        An array X with all 3D images on the dataset.
        An array y with the labels of all 3D images on the dataset.
    """

    labels = sorted(os.listdir(emotions_dir))

    for index1, label in enumerate(labels):
        if index1 == 0:
            print("Recovering arrays for label", label)
            for index2, npy in enumerate(
                    os.listdir(emotions_dir + label)[:neutral_instances]):
                im = np.load(emotions_dir + label + '/' + npy)
                if index1 == 0 and index2 == 0:
                    X = np.zeros((0, im.shape[0], im.shape[1], im.shape[2],
                                  im.shape[3]))
                    y = np.zeros((0, len(labels)))
                X = np.append(X, [im], axis=0)

                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)
        else:
            print("Recovering arrays for label", label)
            for index2, npy in enumerate(os.listdir(emotions_dir + label)):
                im = np.load(emotions_dir + label + '/' + npy)
                X = np.append(X, [im], axis=0)

                y_temp = [0] * len(labels)
                for index, lab in enumerate(labels):
                    if int(label) == int(lab):
                        y_temp[index] = 1.0
                        break
                y = np.append(y, [y_temp], axis=0)

    print("\nShape of X array:", X.shape)
    print("Shape of y array:", y.shape)
    return X, y


def train_test_valid_split(X, y, test_size, valid_size):
    """Generates the train, test and validation datasets.

      Args:
        X: Numpy array with all input images.
        y: Numpy array with all labels.
        test_size: Float percentage in the range (0, 1) of images
            used in test set.
        valid_size: Float percentage in the range (0, 1) of images
            used in validation set.
        
      Returns:
        Arrays of images and labels for each data partition. 
    """

    total_size = len(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size)
    train_size = len(y_train)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train, y_train, test_size=total_size * valid_size / train_size)
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def create_model(input_tensor=None, input_shape=None, classes=7):
    """Creates C3D model with 3 more fully connected layers.

      Args:
        input_tensor: Tensor image to be processed
        input_shape: String tuple (height, width, channels) of input image
        classes: Integer number of classes
        
      Returns:
        Keras neural network model. 
    """

    K.set_image_data_format("channels_last")

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Layer 1
    conv_1 = Convolution3D(
        64, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_1',
        data_format="channels_last")(img_input)
    maxpool_1a = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2), name="maxpool_1a",
        padding='same')(conv_1)
    batchnorm_1 = BatchNormalization(name="batchnorm_1")(maxpool_1a)

    # Layer 2
    conv_2 = Convolution3D(
        128, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_2',
        data_format="channels_last")(batchnorm_1)
    maxpool_2 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2), name="maxpool_2", padding='same')(conv_2)

    # Layer 3
    conv_3 = Convolution3D(
        256, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_3',
        data_format="channels_last")(maxpool_2)

    # Layer 4
    conv_4 = Convolution3D(
        256, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_4',
        data_format="channels_last")(conv_3)
    maxpool_4 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2), name="maxpool_4", padding='same')(conv_4)

    # Layer 5
    conv_5 = Convolution3D(
        512, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_5',
        data_format="channels_last")(maxpool_4)

    # Layer 6
    conv_6 = Convolution3D(
        512, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_6',
        data_format="channels_last")(conv_5)
    maxpool_6 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2), name="maxpool_6", padding='same')(conv_6)

    # Layer 7
    conv_7 = Convolution3D(
        512, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_7',
        data_format="channels_last")(maxpool_6)

    # Layer 8
    conv_8 = Convolution3D(
        512, (3, 3, 3),
        strides=(1, 1, 1),
        padding='same',
        activation='relu',
        name='conv_8',
        data_format="channels_last")(conv_7)
    maxpool_8 = MaxPooling3D(
        (2, 2, 2), strides=(2, 2, 2), name="maxpool_8", padding='same')(conv_8)

    flat = Flatten(name='flatten')(maxpool_6)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    dropout_1 = Dropout(0.33, name='dropout_1')(fc_1)
    net = Dense(classes, activation='softmax', name='predictions')(dropout_1)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, net, name='MOC3D')

    for layer in model.layers:
        layer.trainable = True

    return model


def main(argv):

    emotions_dir = str(argv[0])
    neutral_instances = int(argv[1])
    valid_split = float(argv[2])
    test_split = float(argv[3])
    batch_size = int(argv[4])
    epochs = int(argv[5])

    try:
        assert (valid_split + test_split < .9)
    except AssertionError as e:
        print('Please check the validation and test set sizes.')
        raise

    X, y = recover_3darrays(emotions_dir, neutral_instances=neutral_instances)

    y_counts = np.sum(y, axis=0, dtype=np.int32)
    keys = range(len(y_counts))
    distrib = dict(zip(keys, y_counts))
    print("Class distribution:", distrib)

    X_train, y_train, X_valid, y_valid, X_test, y_test = \
                     train_test_valid_split(X, y, test_split, valid_split)

    print("\n  Training set = ", str(X_train.shape))
    print("Validation set = ", str(X_valid.shape))
    print("      Test set = ", str(X_test.shape) + "\n")

    y_sum = np.append(y_train, y_valid, axis=0)
    y_count = np.sum(y_sum, axis=0, dtype=np.int32)
    y_cnt = np.round(np.max(y_count) / y_count, 4)
    keys = range(len(y_cnt))
    class_weights = dict(zip(keys, y_cnt))

    optimizer = "adam"
    metrics = ['accuracy']
    loss = 'categorical_crossentropy'

    model = create_model(
        input_tensor=None, input_shape=X_train[0].shape, classes=len(keys))

    model.compile(optimizer, loss, metrics)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_acc', factor=0.5, patience=5, min_lr=0.0001, verbose=1)
    early_stopping = EarlyStopping(
        monitor='val_acc',
        min_delta=0,
        patience=10,
        verbose=1,
        mode='max',
        baseline=None)

    history = model.fit(
        X_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=2,
        validation_data=(X_valid, y_valid),
        class_weight=class_weights,
        callbacks=[reduce_lr])

    scores = model.evaluate(X_test, y_test)
    print("Accuracy: %.2f%%" % (scores[1] * 100))

    print("\nDisplaying accuracy curves...")

    # Accuracy Curves
    plt.figure(figsize=[8, 6])
    plt.plot(history.history['acc'], 'r', linewidth=3.0)
    plt.plot(history.history['val_acc'], 'b', linewidth=3.0)
    plt.legend(['Training Accuracy', 'Validation Accuracy'], fontsize=18)
    plt.xlabel('Epochs ', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy Curves', fontsize=16)
    plt.show()

    print("\nSaving model and weights...")

    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights("weigths.h5")
    print("\nModel and weights saved to disk.\n")


if __name__ == "__main__":
    main(sys.argv[1:])
