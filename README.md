[![Build Status](https://travis-ci.org/Nikolay-Lysenko/rl-musician.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/rl-musician)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/rl-musician/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/rl-musician)
[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/rl-musician/maintainability)
[![PyPI version](https://badge.fury.io/py/rl-musician.svg)](https://badge.fury.io/py/rl-musician)

# RL-Musician

## Overview

As of now, this is a proof-of-concept for music composition by reinforcement learning agents. Here, an agent interacts with [piano roll](https://en.wikipedia.org/wiki/Piano_roll#In_digital_audio_workstations) environment, episode end is submission of current piano roll, and reward is determined by applying evaluation rules to the roll.

Comparing to music composition tools such as [MuseNet](https://openai.com/blog/musenet/), reinforcement learning approach (at least in theory) has two advantages:
* Actual creativity lies not in imitation of famous pieces, but in finding new ways to create something to be called art. Reinforcement learning meets this criterion, but supervised learning don't.
* There are [tuning systems](https://en.wikipedia.org/wiki/Musical_tuning#Tuning_systems) other than [equal temperament](https://en.wikipedia.org/wiki/Equal_temperament) (say, in [microtonal music](https://en.wikipedia.org/wiki/Microtonal_music)) and there can be not enough pieces for some of them. A model can not be trained in a supervised manner without a dataset, but, given evaluation rules, an agent can be trained even for an absolutely new tuning system.

Currently, the implementation of environment supports only equal temperament, but this limitation may be eliminated in the future.

## Installation

To install a stable version, run:
```bash
pip install rl-musician
```

## Usage

To train an agent from scratch and to get results produced by it, run:
```bash
python -m rlmusician \
    -c [path/to/your/config] \
    -p [how_long_to_train] \
    -e [how_many_pieces_to_produce]
```

All three arguments are optional. [Default config](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/rlmusician/configs/default_config.yml) is used if `-c` argument is not passed; `-p` and `-e` options have reasonable defaults too.

Created pieces are stored in a directory specified in the config. For each piece, there is a nested directory that contains:
* Piano roll in TSV format;
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format;
* WAV file.
