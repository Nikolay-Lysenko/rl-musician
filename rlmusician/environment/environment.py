"""
Create environment with Gym API.

Author: Nikolay Lysenko
"""


from typing import Any, Dict, List, Tuple

import gym
import numpy as np

from rlmusician.environment.piece import Piece
from rlmusician.environment.evaluation import evaluate
from rlmusician.utils import convert_to_base


class CounterpointEnv(gym.Env):
    """
    An environment where agent sequentially composes melodic lines.
    """

    reward_range = (-np.inf, np.inf)

    def __init__(
            self,
            piece: Piece,
            observation_decay: float,
            reward_for_dead_end: float,
            scoring_coefs: Dict[str, float],
            scoring_fn_params: Dict[str, Dict[str, Any]]
    ):
        """
        Initialize instance.

        :param piece:
            data structure representing musical piece
        :param observation_decay:
            coefficient of exponential decay for previous piano roll's
            time steps
        :param reward_for_dead_end:
            reward for situations where there aren't any allowed actions, but
            piece is not finished
        :param scoring_coefs:
            mapping from scoring function names to their weights in final score
        :param scoring_fn_params:
            mapping from scoring function names to their parameters
        """
        self.piece = piece
        self.observation_decay = observation_decay
        self.reward_for_dead_end = reward_for_dead_end
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params

        self.verbose = False

        n_actions = len(piece.all_movements) ** len(piece.lines)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.action_to_movements = None
        self.__set_action_to_movements()

        roll_height = piece.highest_row_to_show - piece.lowest_row_to_show + 1
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1 / (1 - observation_decay),
            shape=(roll_height,),
            dtype=np.float32
        )

    def __set_action_to_movements(self) -> None:
        """Create mapping from action to line movements."""
        base = len(self.piece.all_movements)
        offset = self.piece.max_skip
        n_lines = len(self.piece.lines)
        action_to_movements = {
            i: [x - offset for x in convert_to_base(i, base, n_lines)]
            for i in range(self.action_space.n)
        }
        self.action_to_movements = action_to_movements

    @property
    def valid_actions(self) -> List[int]:
        """Get actions that are valid at the current step."""
        valid_actions = [
            i for i in range(self.action_space.n)
            if self.piece.check_movements(self.action_to_movements[i])
        ]
        return valid_actions

    @property
    def __decay_coefs(self) -> np.ndarray:
        """Get decay coefficients for passed piano roll's steps."""
        decay_coefs = np.empty((self.piece.last_finished_measure + 1,))
        decay_coefs[0] = 1
        decay_coefs[1:] = self.observation_decay
        decay_coefs = np.cumprod(decay_coefs)
        decay_coefs = np.flip(decay_coefs)
        decay_coefs = decay_coefs.reshape((1, -1))
        return decay_coefs

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Run one step of the environment's dynamics.

        :param action:
            an action provided by an agent to the environment
        :return:
            a tuple of:
            - observation: agent's observation of the current environment,
            - reward: amount of reward returned after previous action,
            - done: whether the episode has ended, in which case further
                    `step()` calls will return undefined results,
            - info: list of next allowed actions
        """
        # Act.
        movements = self.action_to_movements[action]
        self.piece.add_measure(movements)

        # Compute `observation`.
        right_end = self.piece.last_finished_measure + 1
        ready_piano_roll = self.piece.piano_roll[:, :right_end]
        observation = np.sum(self.__decay_coefs * ready_piano_roll, axis=1)

        # Compute `info`.
        info = {'next_actions': self.valid_actions}

        # Compute `done`.
        finish = self.piece.last_finished_measure == self.piece.n_measures - 1
        no_more_actions = len(info['next_actions']) == 0
        done = finish or no_more_actions

        # Compute `reward`.
        if finish:
            reward = evaluate(
                self.piece,
                self.scoring_coefs,
                self.scoring_fn_params,
                self.verbose
            )
        elif no_more_actions:
            reward = self.reward_for_dead_end
        else:
            reward = 0

        return observation, reward, done, info

    def reset(self) -> np.ndarray:
        """
        Reset the state of the environment and return an initial observation.

        :return:
            initial observation
        """
        self.piece.reset()
        initial_observation = self.piece.piano_roll[:, 0]
        return initial_observation

    def render(self, mode='human'):  # pragma: no cover.
        """
        Save piece in various formats.

        :return:
            None
        """
        self.piece.render()
