"""
Define actor model (i.e., model that maps observations to actions).

Author: Nikolay Lysenko
"""


from typing import Tuple

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, Reshape


def create_actor_model(
        observed_roll_shape: Tuple[int, int], n_actions: int
) -> Model:
    """
    Create simple actor network for `PianoRollEnv`.

    :param observed_roll_shape:
        shape of observed part of piano roll
    :param n_actions:
        number of actions available to the agent
    :return:
        actor network which maps observation to action.
    """
    roll_input = Input(shape=observed_roll_shape, name='piano_roll')
    reshaped_input = Reshape(observed_roll_shape + (1,))(roll_input)
    roll_hidden = Conv2D(3, (4, 4), activation='relu')(reshaped_input)
    roll_embedded = Flatten()(roll_hidden)
    output = Dense(n_actions, activation='softmax')(roll_embedded)
    model = Model(inputs=roll_input, outputs=output)
    model.compile(optimizer='sgd', loss='mse')  # Arbitrary unused values.
    return model
