import os
import torch
import torch.nn as nn
from experiment_bp import run


DATASET = 'cifar10'
WHITEN_LVL = 1e-1
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
MOMENTUM = 0.9
WDECAY=5e-4
SCHED_MILESTONES = range(30, 50, 2)
SCHED_GAMMA = 1


if __name__ == '__main__':
	run(
		exp_name=os.path.basename(__file__).rsplit('.', 1)[0],
		dataset=DATASET,
		whiten_lvl=WHITEN_LVL,
		batch_size=BATCH_SIZE,
		epochs=EPOCHS,
	    lr=LR,
		momentum=MOMENTUM,
		wdecay=WDECAY,
		sched_milestones=SCHED_MILESTONES,
		sched_gamma=SCHED_GAMMA)

