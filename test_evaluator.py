from __future__ import print_function

import os
import re

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import matplotlib.cm as mpl_cm
import matplotlib.colors as colors
from matplotlib import ticker

from collections import namedtuple

from typing import List

from comparison_plotter import parse_log_list, LogEntry

TestLogData = namedtuple('TestLogData', 'reward_avg, reward_std, steps_avg,'
                                        'steps_std, n_episodes, n_episodes_solved,'
                                        'name')


def parse_test_log(log_file_path, name):
    reward_avg_pattern = re.compile('.+Testing: reward_avg: (\S+)')
    reward_std_pattern = re.compile('.+Testing: reward_std: (\S+)')
    steps_avg_pattern = re.compile('.+Testing: steps_avg: (\S+)')
    steps_std_pattern = re.compile('.+Testing: steps_std: (\S+)')
    n_episodes_pattern = re.compile('.+Testing: nepisodes: (\S+)')
    n_episodes_solved_pattern = re.compile('.+Testing: nepisodes_solved: (\S+)')

    reward_avg = 0.0
    reward_std = 0.0
    steps_avg = 0.0
    steps_std = 0.0
    n_episodes = 0
    n_episodes_solved = 0

    with open(log_file_path) as log_file:
        for line in log_file.readlines():
            match = reward_avg_pattern.match(line)
            if match:
                reward_avg = float(match.group(1))

            match = reward_std_pattern.match(line)
            if match:
                reward_std = float(match.group(1))

            match = steps_avg_pattern.match(line)
            if match:
                steps_avg = float(match.group(1))

            match = steps_std_pattern.match(line)
            if match:
                steps_std = float(match.group(1))

            match = n_episodes_pattern.match(line)
            if match:
                n_episodes = int(match.group(1))

            match = n_episodes_solved_pattern.match(line)
            if match:
                n_episodes_solved = int(match.group(1))

    return TestLogData(np.array(reward_avg), np.array(reward_std), np.array(steps_avg),
                       np.array(steps_std), n_episodes, n_episodes_solved,
                       name)


def print_test_data(test_entry: LogEntry, test_data: TestLogData):
    # print(
    #     '{0}, {1}, beta = {2}, {name}: episodes = {3}, solved = {4}, reward_avg = {5}, reward_std = {6}, steps_avg = {7}, steps_std = {8}'
    #         .format(
    #         'LSTM' if test_entry.lstm else 'No LSTM',
    #         'ICM' if test_entry.icm else 'No ICM',
    #         test_entry.beta,
    #         test_data.n_episodes,
    #         test_data.n_episodes_solved,
    #         test_data.reward_avg,
    #         test_data.reward_std,
    #         test_data.steps_avg,
    #         test_data.steps_std,
    #         name=test_data.name
    #     )
    # )
    def fmt(fl):
        return '{0:.3f}'.format(fl)

    print(
        ' & '.join([str(x) for x in [test_data.name.replace('_', '\\_'),
                                     'Yes' if test_entry.lstm else 'No',
                                     'Yes' if test_entry.icm else 'No',
                                     '{0:.2f}'.format(test_entry.beta),
                                     fmt(float(test_data.n_episodes_solved) / float(test_data.n_episodes)),
                                     fmt(test_data.reward_avg),
                                     fmt(test_data.reward_std),
                                     fmt(test_data.steps_avg),
                                     fmt(test_data.steps_std) + ' \\\\']])
    )


if __name__ == '__main__':
    log_list_path = 'comparison_log_list.txt'
    logs_path = 'logs'

    log_groups = parse_log_list(log_list_path)
    all_logs = os.listdir(logs_path)

    selected_test_data = []
    for group in log_groups:
        for item in group:
            item_desc = '{0}_{1}'.format(item.machine, item.timestamp)
            item_test_logs = list(filter(lambda x: item_desc in x and 'test' in x, all_logs))
            if len(item_test_logs) > 0:
                test_data = [parse_test_log(os.path.join(logs_path, test_log), test_log[test_log.index('test'):-4]) for
                             test_log in item_test_logs]
                for data in test_data:
                    if data.n_episodes > 50:
                        selected_test_data.append((item, data))


    def name2int(name):
        if name == 'test':
            return 0
        elif name == 'test_gen':
            return 10
        elif name == 'test_gen2':
            return 20
        elif name == 'test_gen3':
            return 30
        else:
            return 100


    selected_test_data = sorted(selected_test_data,
                                key=lambda x: (name2int(x[1].name), int(x[0].lstm), int(x[0].icm), x[0].beta))

    for item, data in selected_test_data:
        print_test_data(item, data)
