[![Build Status](https://travis-ci.org/Nikolay-Lysenko/rl-musician.svg?branch=master)](https://travis-ci.org/Nikolay-Lysenko/rl-musician)
[![codecov](https://codecov.io/gh/Nikolay-Lysenko/rl-musician/branch/master/graph/badge.svg)](https://codecov.io/gh/Nikolay-Lysenko/rl-musician)
[![Maintainability](https://api.codeclimate.com/v1/badges/a43618b5f9454d01186c/maintainability)](https://codeclimate.com/github/Nikolay-Lysenko/rl-musician/maintainability)
[![PyPI version](https://badge.fury.io/py/rl-musician.svg)](https://badge.fury.io/py/rl-musician)

# RL-Musician

## Overview

As of now, this is a proof-of-concept for music composition with reinforcement learning solely. Here, an agent writes [first species counterpoint](https://en.wikipedia.org/wiki/Counterpoint#Species_counterpoint) and environment is based on a special data structure that represents musical piece. An action is filling of current [measure](https://en.wikipedia.org/wiki/Bar_\(music\)) for all melodic lines simultaneously, an episode is finished when all measures are filled one by one, and reward is determined by applying evaluation rules to the resulting piece.

Comparing to music composition tools such as [MuseNet](https://openai.com/blog/musenet/), reinforcement learning approach (at least in theory) has two advantages:
* Actual creativity lies not in imitation of famous pieces, but in finding new ways to create something to be called art. Reinforcement learning meets this criterion, but supervised learning doesn't.
* There are [tuning systems](https://en.wikipedia.org/wiki/Musical_tuning#Tuning_systems) other than [equal temperament](https://en.wikipedia.org/wiki/Equal_temperament) (say, in [microtonal music](https://en.wikipedia.org/wiki/Microtonal_music)) and there can be not enough examples for some of them. A model cannot be trained in a supervised manner without a dataset, but, given some rules, an agent can be trained and evaluated even for an absolutely new tuning system.

To find more details, look at [a draft of a paper](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/docs/paper/paper.pdf).

## Installation

To install a stable version, run:
```bash
pip install rl-musician
```

## Usage

To train an agent from scratch and to get results produced by it, run:
```bash
python -m rlmusician \
    -c [path_to_your_config] \
    -p [how_long_to_train] \
    -e [how_many_pieces_to_produce] \
    -w [path_to_a_file_with_startup_weights]
```

All four arguments are optional. [Default config](https://github.com/Nikolay-Lysenko/rl-musician/blob/master/rlmusician/configs/default_config.yml) is used if `-c` argument is not passed; `-p` and `-e` options have reasonable defaults too. You should skip `-w` option if you have no saved weights from previous launches. If `-w` option is omitted, startup weights are drawn at random with zero mean.  

Training with default settings takes about 30 minutes on a CPU of a regular laptop. If you are on Mac OS, please check that [parallelism is enabled](https://stackoverflow.com/questions/50168647/multiprocessing-causes-python-to-crash-and-gives-an-error-may-have-been-in-progr).

Created pieces are stored in a directory specified in the config. For each piece, there is a nested directory that contains:
* Piano roll in TSV format;
* MIDI file;
* Events file in [sinethesizer](https://github.com/Nikolay-Lysenko/sinethesizer) TSV format;
* WAV file.

Also, weights of the trained agent are saved in a timestamped file.
