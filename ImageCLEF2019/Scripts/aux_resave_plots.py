import os
import json
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def save_logs_as_png(job_dir):
    if not os.path.isfile(os.path.join(job_dir, 'job.json')):
        return

    with open(os.path.join(job_dir, 'job.json')) as f:
        job = json.load(f)

    files = glob(os.path.join(job_dir, 'logdir/*'))
    path = files[0]

    event_acc = EventAccumulator(path)
    event_acc.Reload()

    training_accuracies = event_acc.Scalars('acc')
    validation_accuracies = event_acc.Scalars('val_acc')
    training_loss = event_acc.Scalars('loss')
    validation_loss = event_acc.Scalars('val_loss')

    steps = len(training_accuracies)
    x = np.arange(steps)
    y = np.zeros([steps, 4])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2]
        y[i, 1] = validation_accuracies[i][2]
        y[i, 2] = training_loss[i][2]
        y[i, 3] = validation_loss[i][2]

    plt.plot(x, y[:, 0], label='acc')
    plt.plot(x, y[:, 1], label='val_acc')
    plt.plot(x, y[:, 2], label='loss')
    plt.plot(x, y[:, 3], label='val_loss')
    plt.ylim((0, 1))

    plt.xlabel("Steps")
    plt.ylabel("Values")
    plt.title(job['name'])
    plt.legend(loc='lower left', frameon=True)
    plt.grid()

    min_val_loss = np.min(y[:, 3])

    out_path = os.path.join(job_dir, 'plots_min_val_loss_%.4f.png' % min_val_loss)
    print('Saving ' + out_path)
    plt.savefig(out_path)
    plt.close()


def main():
    dirs = glob('jobs/*-prj*')

    for job_dir in dirs:
        save_logs_as_png(job_dir)


if __name__ == '__main__':
    main()
