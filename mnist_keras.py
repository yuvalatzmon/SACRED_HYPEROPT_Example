import os
import argparse

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import callbacks
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K

import warnings
# see https://stackoverflow.com/a/40846742
warnings.filterwarnings("ignore", message="numpy.dtype size changed")


def get_commandline_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout_rate", type=float, default=0.5)
    parser.add_argument("--fc_dim", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--disable_logging", type=int, default=0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--gpu_memory_fraction", type=float, default=-1,
                        help='GPU fix memory fraction to use, in (0,1], '
                             'or setting -1 for dynamic growth (allow_growth=True).')

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


# Get values for the relevant command line arguments
args, _ = get_commandline_args()


def main(f_log_metrics = lambda logs:None):

    print(vars(args))
    config_tf_session()

    input_shape, output_shape, \
    x_train, y_train, x_val, y_val, x_test, y_test = prepare_data()

    model = get_model(input_shape, output_shape)

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(args.lr),
                  metrics=['accuracy'])

    log_callback = callbacks.LambdaCallback(on_epoch_end=lambda _, logs: f_log_metrics(logs=logs))

    model.fit(x_train, y_train,
              batch_size=args.batch_size,
              epochs=args.epochs,
              verbose=args.verbose,
              validation_data=(x_val, y_val),
              callbacks=[log_callback])
    val_score = model.evaluate(x_val, y_val, verbose=0)
    print('Val loss:', val_score[0])
    print('Val accuracy:', val_score[1])

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])

    print('Saving model: mnist_model.h5')
    model.save('mnist_model.h5')

    return val_score[1], test_score[1]

def config_tf_session():
    gpu_options = tf.GPUOptions(
        per_process_gpu_memory_fraction=args.gpu_memory_fraction, )
    if args.gpu_memory_fraction == -1:
        gpu_options.allow_growth = True
    session_config = tf.ConfigProto(gpu_options=gpu_options)
    sess = tf.Session(config=session_config)

    K.set_session(sess)

def get_model(input_shape, output_shape):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(args.dropout_rate))
    model.add(Flatten())
    model.add(Dense(args.fc_dim, activation='relu'))
    model.add(Dropout(args.dropout_rate))
    model.add(Dense(output_shape, activation='softmax'))

    return model

def prepare_data():
    num_classes = 10

    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train_all, y_train_all), (x_test, y_test) = mnist.load_data()

    # Train, validation & test on small subsets for the sake of this example
    x_train = x_train_all[0:1000, :]
    y_train = y_train_all[0:1000]
    x_val = x_train_all[1000:2000, :]
    y_val = y_train_all[1000:2000]
    x_test = x_test[0:1000, :]
    y_test = y_test[0:1000]

    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_val = x_test.reshape(x_val.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)

    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_val /= 255
    x_test /= 255
    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_val.shape[0], 'val samples')
    print(x_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return input_shape, num_classes, x_train, y_train, x_val, y_val, x_test, y_test
