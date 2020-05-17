[![Build Status](https://travis-ci.org/Nikolay-Lysenko/rl-musician.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/rl-musician)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/rl-musician/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/rl-musician)
[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/rl-musician/maintainability)
[![PyPI version](https://badge.fury.io/py/rl-musician.svg)](https://badge.fury.io/py/rl-musician)

# RL-Musician

## Overview

As of now, this is a proof-of-concept for music composition with reinforcement learning solely. Here, creation of [fifth species counterpoint](https://en.wikipedia.org/wiki/Counterpoint#Species_counterpoint) is considered and environment is based on a special data structure that represents musical piece with pre-defined cantus firmus. An action is adding a new note to a counterpoint line, an episode is finished when counterpoint duration becomes equal to that of cantus firmus, and reward is determined by applying evaluational rules to the resulting piece.

Some pieces generated with this package are uploaded to a publicly available [cloud storage](https://www.dropbox.com/sh/ouv8tcpio8z5flp/AABIx8D-QlIM_FtuQ9LqFntNa?dl=0). A cantus firmus attributed to [Fux](https://en.wikipedia.org/wiki/Johann_Joseph_Fux) is used in all of them.

To find more details, look at [a draft of a paper](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/docs/paper/paper.pdf).

## Installation

To install a stable version, run:
```bash
pip install rl-musician
```

## Usage

To create a reward-maximizing musical piece and some its variations, run:
```bash
python -m rlmusician [-c path_to_your_config]
```

[Default config](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/rlmusician/configs/default_config.yml) is used if `-c` argument is not passed. Search of optimal piece with these default settings takes about 30 minutes on a CPU of a regular laptop. Before creating a new config, it might be useful to look at [an example with explanations](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/docs/config_with_explanations.yml).

If you are on Mac OS, please check that [parallelism is enabled](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).

Generated pieces are stored in a directory specified in the config. For each piece, there is a nested directory that contains:
* Piano roll in TSV format;
* MIDI file;
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format;
* WAV file.
