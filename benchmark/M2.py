# -*- coding: utf-8 -*-
"""
Compute M2
"""

import numpy as np

target_fname = './results/R3_exlarge_gpomd'

base_fname = './results/R3_exlarge_simple001'

target = np.load(target_fname+'.npy')
base = np.load(base_fname+'.npy')

assert len(target) == len(base)

print('Sanity Check' + target_fname)
print(np.mean(target), np.std(target))

improvements = []
for i in range(len(target)):
    impr = 100.0 * (target[i] - base[i])/base[i]
    improvements.append(impr)

print('Improvements over baseline methods')
print('Mean: {}, Std: {}'.format(np.mean(improvements), np.std(improvements)))
