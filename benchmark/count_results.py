# -*- coding: utf-8 -*-
"""
Compute statistical results
"""

import numpy as np

folder = '../eval/R3/ExLarge1207/md02_2500'
print(folder)
save = True
save_to = 'R3_exlarge_gpomd'
#save_to = 'tmp'

start_no = 1
end_no = 100
total_no = end_no - start_no + 1
batch_size = 10

rewards = []
avails = []

for prob_no in range(start_no, end_no+1):
    save_folder_prob = folder + '/%05d' % prob_no
    hourly_r_list = []
    hourly_a_list = []    
    
    for i in range(batch_size):
        fname = save_folder_prob + '/%03d' % i

        hourly_r_list.append(np.mean(np.load(fname+'.npy')))
        hourly_a_list.append(np.mean(np.load(fname+'_avail.npy')))

    rewards.append(np.mean(hourly_r_list))
    avails.append(np.mean(hourly_a_list))

print('Hourly Rewards')
print('Mean: {}, Std: {}'.format(np.mean(rewards), np.std(rewards)))
print('Max: {}'.format(np.max(rewards)))
print('Plane Availability')
print('Mean: {}, Std: {}'.format(np.mean(avails), np.std(avails)))
print('Max: {}'.format(np.max(avails)))

if save:
    np.save(save_to, rewards)
