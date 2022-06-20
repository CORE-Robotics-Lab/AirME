# -*- coding: utf-8 -*-
"""
data generation for evaluating different algorithms
"""

import os
import random
import pickle

from utils import RepairEnv, get_default_param

folder = './gen/large'
start_no = 1
end_no = 100
total_no = end_no - start_no + 1

random_init = True
percent_broken = 0.1

if not os.path.exists(folder):
    os.makedirs(folder)

for prob_no in range(start_no, end_no+1):
    fname = folder + '/%05d_env.pkl' % prob_no
    
    # sample plane/crew size
    # small
    # num_airliners = random.randint(16, 24)
    # num_helicopters = random.randint(8, 12)
    # num_crews = random.randint(6, 8)    

    # medium
    # num_airliners = random.randint(32, 48)
    # num_helicopters = random.randint(16, 24)
    # num_crews = random.randint(12, 16)
    
    # large
    num_airliners = random.randint(64, 96)
    num_helicopters = random.randint(32, 48)
    num_crews = random.randint(24, 32)
    
    plane_info = get_default_param(num_airliners, num_helicopters, random_init,
                                   percent_broken)        
    # initialize env
    r = RepairEnv(num_airliners, num_helicopters,
                  plane_info['num_parts_list'], plane_info['scale_lists'], 
                  plane_info['shape_lists'], plane_info['prob_uses'], plane_info['hour_scales'],
                  plane_info['hours_list'], plane_info['num_landings_list'],
                  plane_info['broken_list'], plane_info['reward_list'],
                  num_crews)    
    
    # save env as pickle file
    with open(fname, 'wb') as f:
        pickle.dump(r, f)    
    
print('Generated {} problem instances under {}'.format(total_no, folder))
