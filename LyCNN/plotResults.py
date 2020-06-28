
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

import argparse

parser = argparse.ArgumentParser(description='Takes directory to trained model')
parser.add_argument('--model', type=str, nargs = 1, help ='path to trained model', required = False)
args = parser.parse_args()
modelPath = args.model[0]

results = pd.read_json(modelPath+'/stat.json')

loss = results[['epoch_num','cross_entropy_loss']]

epoch = loss[['epoch_num']].values.flatten()
cross_entropy_loss = loss[['cross_entropy_loss']].values.flatten()
print(epoch)

fig = plt.figure()
loss.plot(x='epoch_num', y='cross_entropy_loss')

plt.show()
fig.savefig('../output/epoch-400d1-9d2-3d3-1_randcrp.png')
