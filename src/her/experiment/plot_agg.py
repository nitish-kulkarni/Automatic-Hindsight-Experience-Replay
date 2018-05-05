import os
import matplotlib.pyplot as plt
import numpy as np
import json
import seaborn as sns; sns.set()
import glob2
import argparse

def smooth_reward_curve(x, y):
    halfwidth = int(np.ceil(len(x) / 60))  # Halfwidth of our smoothing convolution
    k = halfwidth
    xsmoo = x
    ysmoo = np.convolve(y, np.ones(2 * k + 1), mode='same') / np.convolve(np.ones_like(y), np.ones(2 * k + 1),
        mode='same')
    return xsmoo, ysmoo


def load_results(file):
    if not os.path.exists(file):
        return None
    with open(file, 'r') as f:
        lines = [line for line in f]
    if len(lines) < 2:
        return None
    keys = [name.strip() for name in lines[0].split(',')]
    data = np.genfromtxt(file, delimiter=',', skip_header=1, filling_values=0.)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    assert data.ndim == 2
    assert data.shape[-1] == len(keys)
    result = {}
    for idx, key in enumerate(keys):
        result[key] = data[:, idx]
    return result



parser = argparse.ArgumentParser()
parser.add_argument('dir', type=str)
#parser.add_argument('--res-dir','',type=str)
parser.add_argument('--smooth', type=int, default=1)
args = parser.parse_args()

# Load all data.
data = {}
print(args.dir)
paths = [os.path.abspath(os.path.join(path, '..')) for path in glob2.glob(os.path.join(args.dir, '**', 'progress.csv'))]
for curr_path in paths:
    if not os.path.isdir(curr_path) or 'GoalGen' in curr_path:
        print('skipping {}'.format(curr_path))
        continue
    results = load_results(os.path.join(curr_path, 'progress.csv'))
    if not results:
        print('skipping {}'.format(curr_path))
        continue
    print('loading {} ({})'.format(curr_path, len(results['epoch'])))
    with open(os.path.join(curr_path, 'params.json'), 'r') as f:
        params = json.load(f)

    success_rate = np.array(results['test/success_rate'])
    epoch = np.array(results['epoch']) + 1
    env_id = params['env_name']
    replay_strategy = params['replay_strategy']

    if replay_strategy == 'best_k':
        x = params['gg_k']
        y_mean = np.mean(success_rate)
        y_max = np.max(success_rate)
        yerr = np.std(success_rate)
    # Process and smooth data.
        assert success_rate.shape == epoch.shape
        
        if env_id not in data:
            data[env_id] = []
        data[env_id].append((x, y_mean,yerr,y_max))

# Plot data.
for env_id in sorted(data.keys()):
    print('exporting {} mean info'.format(env_id))
    plt.clf()

    data[env_id] = sorted(data[env_id], key=lambda tup: tup[0])

    xs = [ x[0] for x in data[env_id]]
    ys = [ x[1] for x in data[env_id]]
    err = [ x[2] for x in data[env_id]]
    y_maxs = [ x[3] for x in data[env_id]]

    plt.errorbar(xs,ys,yerr=err,marker='o', linestyle='-', capsize=2, elinewidth=0.5)
    plt.title(env_id)
    plt.xlabel('k')
    plt.ylabel('Success Rate')
    plt.legend()
    plt.savefig(os.path.join(args.dir, 'fig_mean_{}.pdf'.format(env_id)))

    print('exporting {} max info'.format(env_id))
    plt.clf()
    plt.plot(xs,y_maxs,marker='o')
    plt.title(env_id)
    plt.xlabel('k')
    plt.ylabel('Max Success Rate')
    plt.legend()
    plt.savefig(os.path.join(args.dir, 'fig_max_{}.pdf'.format(env_id)))
