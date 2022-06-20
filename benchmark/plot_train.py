# -*- coding: utf-8 -*-
"""
Ablation studies
    Compare training curves / hourly
"""

import torch

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.rc('xtick', labelsize=20)     
matplotlib.rc('ytick', labelsize=20)

plt.figure(figsize=(12, 8))

total_points = 350
x = list(range(total_points))
smooth = True
kernel_size = 5


gposd_list = ['../sd_abl01/checkpoint_00400.tar',
              '../sd_abl02/checkpoint_00410.tar',
              '../sd_abl03/checkpoint_00410.tar',
              '../sd_abl04/checkpoint_00380.tar']
gpovalue_list = ['../value01/checkpoint_00510.tar',
                 '../value02/checkpoint_01050.tar',
                 '../value03/checkpoint_00360.tar',
                 '../value04/checkpoint_00360.tar']


'''
Plot gposd
'''
for i in range(len(gposd_list)):
    cp = torch.load(gposd_list[i])
    hourly_rs = np.array(cp['hourly_rs'])
    hourly_std = np.array(cp['hourly_std'])

    if smooth:
        kernel = np.ones(kernel_size) / kernel_size
        hourly_rs = np.convolve(hourly_rs, kernel, mode='same')
    
    plt.plot(x, hourly_rs[:total_points], label = 'GPO-time%03d' % i)
    plt.fill_between(x, hourly_rs[:total_points] - hourly_std[:total_points],
                     hourly_rs[:total_points] + hourly_std[:total_points], alpha=0.4)


'''
Plot gpovalue
'''
for i in range(len(gpovalue_list)):
    cp = torch.load(gpovalue_list[i])
    hourly_rs = np.array(cp['hourly_rs'])
    hourly_std = np.array(cp['hourly_std'])

    if smooth:
        kernel = np.ones(kernel_size) / kernel_size
        hourly_rs = np.convolve(hourly_rs, kernel, mode='same')
    
    plt.plot(x, hourly_rs[:total_points], label = 'GPO-value%03d' % i)
    plt.fill_between(x, hourly_rs[:total_points] - hourly_std[:total_points],
                     hourly_rs[:total_points] + hourly_std[:total_points], alpha=0.4)


# plt.ylim(bottom=0)
plt.xlim([0, total_points])
plt.xlabel('No. of Training Epochs', fontsize=20)
plt.ylabel('Normalized Hourly Reward', fontsize=20)
plt.legend()
#plt.legend(['PG-time-no fix', 'PG-time-fix', 'A2C-GAE'], fontsize=20)
plt.grid(linestyle='--')
# plt.plot(x, [0.535185925 for i in range(total_points)])

plt.tight_layout()
plt.savefig('./comp.png')
