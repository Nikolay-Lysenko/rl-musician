"""
Define some versions of agents.

Author: Nikolay Lysenko
"""


from typing import Tuple

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input, concatenate
from keras.optimizers import Adam
from rl.agents.dqn import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from rl.processors import MultiInputProcessor


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
    curr_score_input = Input(shape=(1, 1), name='current_score')
    curr_score_flatten = Flatten()(curr_score_input)
    hidden = concatenate([roll_embedded, curr_score_flatten])
    output = Dense(n_actions, activation='softmax')(hidden)
    model = Model(inputs=[roll_input, curr_score_input], outputs=output)
    return model


def create_dqn_agent(
        observed_roll_shape: Tuple[int, int], n_actions: int
) -> DQNAgent:
    """
    Create simple agent.

    :param observed_roll_shape:
        shape of observed part of piano roll
    :param n_actions:
        number of actions available to the agent
    :return:
        DQN agent
    """
    memory = SequentialMemory(limit=50000, window_length=1)
    policy = BoltzmannQPolicy()
    agent = DQNAgent(
        model=create_actor_network(observed_roll_shape, n_actions),
        nb_actions=n_actions,
        processor=MultiInputProcessor(2),
        memory=memory,
        policy=policy,
        nb_steps_warmup=2000,
        target_model_update=1e-2
    )
    agent.compile(Adam(lr=1e-3), metrics=['mae'])
    return agent
