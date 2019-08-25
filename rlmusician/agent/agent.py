"""
Define some versions of agents.

Author: Nikolay Lysenko
"""


from typing import Tuple

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input
from rl.agents.cem import CEMAgent
from rl.memory import EpisodeParameterMemory


def create_actor_network(
        observed_roll_shape: Tuple[int, int], n_actions: int
) -> Model:
    """
    Create simple actor network for `MusicCompositionEnv`.

    :param observed_roll_shape:
        shape of observed part of piano roll
    :param n_actions:
        number of actions available to the agent
    :return:
        actor network which maps observation to action.
    """
    roll_input = Input(shape=(1,) + observed_roll_shape, name='piano_roll')
    roll_hidden = Conv2D(3, (4, 4), activation='relu', data_format='channels_first')(roll_input)
    roll_embedded = Flatten()(roll_hidden)
    output = Dense(n_actions, activation='softmax')(roll_embedded)
    model = Model(inputs=[roll_input], outputs=output)
    return model


def create_cem_agent(
        observed_roll_shape: Tuple[int, int], n_actions: int
) -> CEMAgent:
    """
    Create simple agent.

    :param observed_roll_shape:
        shape of observed part of piano roll
    :param n_actions:
        number of actions available to the agent
    :return:
        agent of Cross-Entropy Method (CEM)
    """
    agent = CEMAgent(
        model=create_actor_network(observed_roll_shape, n_actions),
        nb_actions=n_actions,
        memory=EpisodeParameterMemory(limit=50000, window_length=1),
        batch_size=200,
        nb_steps_warmup=50000,
        train_interval=50,
        elite_frac=0.2
    )
    agent.compile()
    return agent
