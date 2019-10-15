"""
Define actor model (i.e., model that maps observation to action probabilities).

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


def create_actor_model(
        observation_shape: Tuple[int, int], n_actions: int
) -> 'keras.models.Model':
    """
    Create simple actor network for `PianoRollEnv`.

    :param observation_shape:
        shape of observation
    :param n_actions:
        number of actions available to the agent
    :return:
        actor network that maps observation to action
    """
    # Here, all `keras`-related stuff is imported by child processes only.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        import_keras_silently()
        from keras.models import Model
        from keras.layers import Dense, Input
        mute_tensorflow()

    observed_input = Input(shape=observation_shape, name='input')
    output = Dense(n_actions, activation='softmax')(observed_input)
    model = Model(inputs=observed_input, outputs=output)
    model.compile(optimizer='sgd', loss='mse')  # Arbitrary unused values.
    return model
