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
    An environment where counterpoint line is composed given cantus firmus.
    """

    reward_range = (-np.inf, np.inf)
    valid_durations = [1, 2, 4, 8]

    def __init__(
            self,
            piece: Piece,
            scoring_coefs: Dict[str, float],
            scoring_fn_params: Dict[str, Dict[str, Any]],
            reward_for_dead_end: float,
            verbose: bool = False
    ):
        """
        Initialize instance.

        :param piece:
            data structure representing musical piece with florid counterpoint
        :param scoring_coefs:
            mapping from scoring function names to their weights in final score
        :param scoring_fn_params:
            mapping from scoring function names to their parameters
        :param reward_for_dead_end:
            reward for situations where there aren't any allowed actions, but
            piece is not finished
        :param verbose:
            if it is set to `True`, breakdown of reward is printed at episode
            end
        """
        self.piece = piece
        self.scoring_coefs = scoring_coefs
        self.scoring_fn_params = scoring_fn_params
        self.reward_for_dead_end = reward_for_dead_end
        self.verbose = verbose

        n_actions = len(piece.all_movements) * len(self.valid_durations)
        self.action_space = gym.spaces.Discrete(n_actions)
        self.action_to_line_continuation = None
        self.__set_action_to_line_continuation()

        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=self.piece.piano_roll.shape,
            dtype=np.int32
        )

    def __set_action_to_line_continuation(self) -> None:
        """Create mapping from action to a pair of movement and duration."""
        base = len(self.piece.all_movements)
        required_len = 2
        raw_mapping = {
            action: (x for x in convert_to_base(action, base, required_len))
            for action in range(self.action_space.n)
        }
        offset = self.piece.max_skip
        action_to_continuation = {
            action: (movement_id - offset, 2 ** duration_id)
            for action, (duration_id, movement_id) in raw_mapping.items()
        }
        self.action_to_line_continuation = action_to_continuation

    @property
    def valid_actions(self) -> List[int]:
        """Get actions that are valid at the current step."""
        valid_actions = [
            i for i in range(self.action_space.n)
            if self.piece.check_validity(*self.action_to_line_continuation[i])
        ]
        return valid_actions

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
        movement, duration = self.action_to_line_continuation[action]
        self.piece.add_line_element(movement, duration)

        observation = self.piece.piano_roll
        info = {'next_actions': self.valid_actions}

        past_duration = self.piece.current_time_in_eighths
        piece_duration = self.piece.total_duration_in_eighths
        finished = past_duration == piece_duration
        no_more_actions = len(info['next_actions']) == 0
        done = finished or no_more_actions

        if finished:
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
        initial_observation = self.piece.piano_roll
        return initial_observation

    def render(self, mode='human'):  # pragma: no cover.
        """
        Save piece in various formats.

        :return:
            None
        """
        self.piece.render()
