from __future__ import print_function

import os
import re

import numpy as np
import matplotlib.pyplot as plt

from collections import namedtuple

LogEntry = namedtuple('LogEntry', 'machine, timestamp, icm, lstm, beta, number')


def parse_log_list(log_list_path):
    log_groups = []
    with open(log_list_path) as log_list_file:
        group = []
        for line in log_list_file.readlines():
            trimmed = line.rstrip()

            if trimmed == '':
                if len(group) > 0:
                    log_groups.append(group)
                group = []
            else:
                parts = [p.strip() for p in trimmed[8:].split(',')]
                machine = parts[0].split('_')[0]
                timestamp = parts[0].split('_')[1]
                icm = parts[3].split(' ')[1] == 'full' or parts[3].split(' ')[1] == 'on'
                lstm = parts[4] == 'LSTM'
                beta = float(parts[6].split(' ')[1])
                number = 1
                if len(parts) == 8 and len(parts[7]) > 0:
                    number = int(re.sub(r"\D", "", parts[7].split(' ')[0]))
                group.append(LogEntry(machine, timestamp, icm, lstm, beta, number))
    return log_groups


def parse_log(log_file_path):
    reward_avg_pattern = re.compile('.+Iteration: (\d+); reward_avg: (\S+)')
    steps_avg_pattern = re.compile('.+Iteration: (\d+); steps_avg: (\S+)')

    timesteps = []
    reward_avg = []
    steps_avg = []

    with open(log_file_path) as log_file:
        for line in log_file.readlines():
            match = reward_avg_pattern.match(line)
            if match:
                timesteps.append(int(match.group(1)))
                reward_avg.append(float(match.group(2)))
            match = steps_avg_pattern.match(line)
            if match:
                steps_avg.append(float(match.group(2)))

    assert len(timesteps) == len(reward_avg) and len(timesteps) == len(steps_avg)

    return timesteps, reward_avg, steps_avg


def plot_group(group_res, group_entries):
    def find_nearest(array, value):
        idx = (np.abs(array - value)).argmin()
        return array[idx]

    lengths = [len(t[0]) for t in group_res]
    reference_run_idx = np.argmin(lengths)
    reference_timesteps = group_res[reference_run_idx][0]

    other_run_idxs = list(range(len(group_res)))
    other_run_idxs.remove(reference_run_idx)

    closest = []
    for idx in other_run_idxs:
        other_closest = []
        other_timesteps = np.array(group_res[idx][0])
        for timestep in reference_timesteps:
            other_closest.append(find_nearest(other_timesteps, timestep))
        closest.append(other_closest)

    print(reference_timesteps)
    for steps in closest:
        print(steps)
    print('')


if __name__ == '__main__':
    log_list_path = 'comparison_log_list.txt'

    log_groups = parse_log_list(log_list_path)

    for group in log_groups:
        group_res = []
        for entry in group:
            timesteps, reward_avg, steps_avg = parse_log(
                os.path.join('logs', '{0}_{1}.log'.format(entry.machine, entry.timestamp))
            )
            group_res.append((timesteps, reward_avg, steps_avg))
        plot_group(group_res, group)
