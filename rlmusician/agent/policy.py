"""
Define a policy.

Here, a policy maps a pair of an observation and an encoded action to
score of this action.

Author: Nikolay Lysenko
"""


from typing import Tuple


def mute_tensorflow() -> None:
    """
    Mute numerous warnings from `tensorflow`, because they clutter up `stdout`.

    :return:
        None
    """
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    import tensorflow as tf
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


def import_keras_silently() -> None:
    """
    Prevent printing 'Using TensorFlow backend' every time a process starts.

    :return:
        None
    """
    import os
    import sys
    stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    import keras
    sys.stderr = stderr


def create_policy(
        input_shape: Tuple[int], hidden_layer_size: int
) -> 'keras.models.Model':
    """
    Create simple policy for `CounterpointEnv`.

    :param input_shape:
        shape of input without the first dimension (reserved for batch size)
    :param hidden_layer_size:
        number of hidden units
    :return:
        policy as a neural network with one hidden layer
    """
    # Avoid mandatory importing `keras`-related stuff in the main process.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import_keras_silently()
        from keras.models import Model
        from keras.layers import Dense, Input
        mute_tensorflow()

    inp = Input(shape=input_shape, name='input')
    hidden = Dense(hidden_layer_size, activation='relu', name='hidden')(inp)
    output = Dense(1)(hidden)
    model = Model(inputs=inp, outputs=output)
    model.compile(optimizer='sgd', loss='mse')  # Arbitrary unused values.
    return model
