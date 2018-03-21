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

from scipy.signal import savgol_filter

LogEntry = namedtuple('LogEntry', 'machine, timestamp, icm, lstm, beta, number')
LogData = namedtuple('LogData', 'timesteps, reward_avg, reward_std, steps_avg, '
                                'steps_std, n_episodes, action_counts')


def combine_2_means_stds(mean1, mean2, std1, std2, n1, n2):
    mean = (n1 * mean1 + n2 * mean2) / (n1 + n2)
    return mean, np.sqrt(
        (n1 * std1 ** 2 + n2 * std2 ** 2 + n1 * (mean1 - mean) ** 2 + n2 * (mean2 - mean) ** 2) / (n1 + n2))


def combine_3_means_stds(mean1, mean2, mean3, std1, std2, std3, n1, n2, n3):
    mean = (n1 * mean1 + n2 * mean2 + n3 * mean3) / (n1 + n2 + n3)
    return mean, np.sqrt(
        (n1 * std1 ** 2 + n2 * std2 ** 2 + n3 * std3 ** 2 +
         n1 * (mean1 - mean) ** 2 + n2 * (mean2 - mean) ** 2 + n3 * (mean3 - mean) ** 2) / (n1 + n2 + n3))


def plot_mean_and_confidence_interval(ax, x, mean, lb, ub, label=None, alpha=0.5,
                                      color_mean=None, color_shading=None):
    # plot the shaded range of the confidence intervals
    # ax.fill_between(x, ub, lb,
    #                 color=color_shading, alpha=.5)
    ax.fill_between(x, ub, lb, alpha=alpha, edgecolor='gray')
    # plot the mean on top
    # mean = savgol_filter(mean, 15, 4)
    ax.plot(x, mean, label=label)


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
        if len(group) > 0:
            log_groups.append(group)
        group = []
    return log_groups


def parse_log(log_file_path):
    reward_avg_pattern = re.compile('.+Iteration: (\d+); reward_avg: (\S+)')
    reward_std_pattern = re.compile('.+Iteration: (\d+); reward_std: (\S+)')
    steps_avg_pattern = re.compile('.+Iteration: (\d+); steps_avg: (\S+)')
    steps_std_pattern = re.compile('.+Iteration: (\d+); steps_std: (\S+)')
    n_episodes_pattern = re.compile('.+Iteration: (\d+); nepisodes: (\S+)')
    action_counts_pattern = re.compile('.+Iteration: (\d+); action_counts: \[(.+)\]')

    timesteps = []
    reward_avg = []
    reward_std = []
    steps_avg = []
    steps_std = []
    n_episodes = []
    action_counts = []

    with open(log_file_path) as log_file:
        for line in log_file.readlines():
            match = reward_avg_pattern.match(line)
            if match:
                timesteps.append(int(match.group(1)))
                reward_avg.append(float(match.group(2)))

            match = reward_std_pattern.match(line)
            if match:
                reward_std.append(float(match.group(2)))

            match = steps_avg_pattern.match(line)
            if match:
                steps_avg.append(float(match.group(2)))

            match = steps_std_pattern.match(line)
            if match:
                steps_std.append(float(match.group(2)))

            match = n_episodes_pattern.match(line)
            if match:
                n_episodes.append(int(match.group(2)))

            match = action_counts_pattern.match(line)
            if match:
                action_counts.append(np.fromstring(match.group(2), sep=' '))

    assert len(timesteps) == len(reward_avg) and len(timesteps) == len(steps_avg)

    return LogData(np.array(timesteps), np.array(reward_avg), np.array(reward_std), np.array(steps_avg),
                   np.array(steps_std), np.array(n_episodes), np.array(action_counts))


# TODO: improve function name and local vars names
def fix_logs_timesteps(group_res: List[LogData], verbose: bool = False) -> List[LogData]:
    def find_idx_nearest(array, value):
        return (np.abs(array - value)).argmin()

    lengths = [len(t[0]) for t in group_res]
    reference_run_idx = int(np.argmin(lengths))
    reference_timesteps = group_res[reference_run_idx].timesteps

    other_run_idxs = list(range(len(group_res)))
    other_run_idxs.remove(reference_run_idx)

    closest_idx = []
    for idx in other_run_idxs:
        other_closest = []
        for timestep in reference_timesteps:
            other_closest.append(find_idx_nearest(group_res[idx].timesteps, timestep))
        closest_idx.append(other_closest)
    fixed_group_res = [LogData] * len(group_res)
    fixed_group_res[reference_run_idx] = group_res[reference_run_idx]

    for i, idx in enumerate(other_run_idxs):
        res = group_res[idx]
        cl_i = closest_idx[i]
        fixed_group_res[idx] = \
            LogData(
                res.timesteps[cl_i], res.reward_avg[cl_i], res.reward_std[cl_i], res.steps_avg[cl_i],
                res.steps_std[cl_i], res.n_episodes[cl_i], res.action_counts[cl_i]
            )
    if verbose:
        diff = [fixed_group_res[reference_run_idx].timesteps - e.timesteps for e in fixed_group_res]
        print(np.max(diff, axis=0), np.max(diff, axis=0).sum())
    return fixed_group_res


def plot_group(ax, group_res: List[LogData], group_entries: List[LogEntry],
               to_plot, alpha, lstm_in_label=False):
    group_res = fix_logs_timesteps(group_res)

    n = np.sum([res.n_episodes for res in group_res], axis=0)
    mean, std = combine_3_means_stds(
        getattr(group_res[0], to_plot + '_avg'),
        getattr(group_res[1], to_plot + '_avg'),
        getattr(group_res[2], to_plot + '_avg'),
        getattr(group_res[0], to_plot + '_std'),
        getattr(group_res[1], to_plot + '_std'),
        getattr(group_res[2], to_plot + '_std'),
        group_res[0].n_episodes,
        group_res[1].n_episodes,
        group_res[2].n_episodes
    )

    lower_bound, upper_bound = st.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
    # lower_bound, upper_bound = mean - std, mean + std

    label = r'{0}{1}, $\beta={2}$'.format(
        ('LSTM, ' if group_entries[0].lstm else 'No LSTM, ') if lstm_in_label else '',
        'ICM' if group_entries[0].icm else 'No ICM',
        group_entries[0].beta
    )

    def exploration_desc(entry):
        if entry.beta > 0:
            return 'ICM+Entropy' if entry.icm else 'Entropy'
        else:
            return 'ICM' if entry.icm else 'None'

    label = '{0}{1}'.format(
        ('LSTM+' if group_entries[0].lstm else 'No LSTM+') if lstm_in_label else '',
        exploration_desc(group_entries[0])
    )

    plot_mean_and_confidence_interval(
        ax, group_res[0].timesteps, mean, lower_bound, upper_bound,
        label=label,
        alpha=alpha, color_mean='b', color_shading='b'
    )


def plot_statistic(data_groups, to_plot, y_label, y_lim, legend_loc,
                   save_path):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.set_prop_cycle(
        color=['xkcd:bright lavender', 'xkcd:cobalt blue', 'xkcd:lightish red'],
        dashes=[[5, 1], [1, 1], [1, 0]],  # [1, 5], [5, 5], [3, 5, 1, 5]],
        linewidth=[1.5, 1.5, 2.5]
    )
    fill_alpha = [0.25, 0.25, 0.5]
    ax.set_ylabel(y_label, fontsize=18)
    ax.set_xlabel('Training steps', fontsize=18)
    ax.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))

    for i, group in enumerate(data_groups):
        group_res = []
        for entry in group:
            group_res.append(parse_log(
                os.path.join('logs', '{0}_{1}.log'.format(entry.machine, entry.timestamp))
            ))
        plot_group(ax, group_res, group, to_plot, fill_alpha[i % len(fill_alpha)])
    plt.legend(loc=legend_loc, handlelength=3, fontsize=18)
    ax.set_ylim(y_lim)
    ax.set_xlim([0, 3000000])
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    ax.xaxis.offsetText.set_fontsize(14)
    ax.yaxis.offsetText.set_fontsize(14)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    plt.savefig(os.path.join(save_path, '{0}.pdf'.format(to_plot)),
                bbox_inches='tight', pad_inches=0.05, dpi=300)
    plt.clf()


# TODO: improve data flow, read only once here, not inside plot_statistic
if __name__ == '__main__':
    log_list_path = 'comparison_log_list.txt'

    log_groups = parse_log_list(log_list_path)

    # TODO: improve this part
    for group in log_groups:
        group_res = []
        for entry in group:
            group_res.append(parse_log(
                os.path.join('logs', '{0}_{1}.log'.format(entry.machine, entry.timestamp))
            ))
        print([(np.max(l.reward_avg), l.timesteps[np.argmax(l.reward_avg)]) for l in group_res])

    lstm_groups = list(filter(lambda x: x[0].lstm, log_groups))
    no_lstm_groups = list(filter(lambda x: not x[0].lstm, log_groups))

    plt.rc('font', family='Times New Roman')

    plot_statistic(lstm_groups, to_plot='reward', y_label='Average reward',
                   y_lim=[-50, 5], legend_loc=4, save_path='figs/lstm')
    plot_statistic(lstm_groups, to_plot='steps', y_label='Average steps',
                   y_lim=[0, 1000], legend_loc=1, save_path='figs/lstm')

    plot_statistic(no_lstm_groups, to_plot='reward', y_label='Average reward',
                   y_lim=[-50, 5], legend_loc=4, save_path='figs/no lstm')
    plot_statistic(no_lstm_groups, to_plot='steps', y_label='Average steps',
                   y_lim=[0, 3000], legend_loc=1, save_path='figs/no lstm')
