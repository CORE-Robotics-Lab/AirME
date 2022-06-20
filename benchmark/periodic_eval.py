# -*- coding: utf-8 -*-
"""
Evaluate in a loop / plot saved in a folder without showing

Multi-Action version
    Multiple scheduling decisions for one time step
    Changeable plane/crew number

load pre-initialized env from local files

Evaluate on N env files: total_no
    run M times on each file and take the mean: batch_size
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle

sys.path.append('../')

# from utils import RepairEnv, get_default_param
from heuristics import PeriodicScheduler

'''
Perform multiple runs using the same env
Inputs:
    fname: env file
    save_folder_prob: save folder for current env
    norm_r: boolen
    simulation_time: int
    batch_size: how many runs to perfrom
    interval: scheduling interval
Return the mean values
'''
def run_test(fname, save_folder_prob, norm_r, simulation_time, batch_size, interval, Rtype):
    hourly_r_list = []
    hourly_a_list = []
    
    for i in range(batch_size):
        print('Batch {}/{}.'.format(i+1, batch_size))
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        
        scheduler = PeriodicScheduler(r.num_planes, r.num_crews, interval)
        
        time_t = []
        total_reward = []
        plane_avail = []        
        
        for t in range(simulation_time):
            crews_avail = r.get_available_crews()
            planes_broken, planes_avail = r.get_broken_and_available()
            crew_list, plane_list = scheduler.get_schedule_actions(crews_avail,
                                                            planes_broken,
                                                            planes_avail,
                                                            t)

            cost, reward, avail_count = r.step_multi(crew_list, plane_list, 
                                                     verbose = False, norm_r = norm_r)
        
            time_t.append(r.total_hours)
            
            if Rtype == 'R1':
                total_reward.append(reward-cost)
            elif Rtype == 'R2':
                total_reward.append(reward)
            elif Rtype == 'R3':
                R3_reward = avail_count / r.num_planes
                total_reward.append(R3_reward)
            else:
                print('Error')              
            
            plane_avail.append(avail_count / r.num_planes)

        savefname_i = save_folder_prob + '/%03d' % i
        print('Saving to '+savefname_i)
        plt.figure(figsize=(15,6))
        plt.plot(time_t, np.array(total_reward),'b')
        plt.xlabel('Time (Hour)')
        plt.ylabel('Reward')
        plt.savefig(savefname_i+'.png')
        plt.close()
       
        np.save(savefname_i, total_reward)
        np.save(savefname_i+'_avail', plane_avail)
        
        hourly_r = np.mean(np.array(total_reward))
        hourly_r_list.append(hourly_r)
        print('hourly_r', hourly_r)
        
        hourly_a = np.mean(np.array(plane_avail))
        hourly_a_list.append(hourly_a)
        print('hourly_a', hourly_a)
    
    return np.mean(hourly_r_list), np.mean(hourly_a_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-folder', type=str)
    parser.add_argument('--save-folder', type=str)
    parser.add_argument('--start-no', default=1, type=int)
    parser.add_argument('--end-no', default=100, type=int)
    parser.add_argument('--sim-time', default=720, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--interval', default=1, type=int)
    parser.add_argument('--Rtype', type=str)
    args = parser.parse_args()

    simulation_time = args.sim_time
    norm_r = True
    interval = args.interval
    token = '%03d' % interval

    # Rtype: reward type
    Rtype = args.Rtype    
    assert Rtype in ['R1', 'R2', 'R3']
    print(Rtype + ' reward')
        
    data_folder = args.data_folder
    save_folder = args.save_folder + '/periodic' + token
    
    start_no = args.start_no
    end_no = args.end_no
    total_no = end_no - start_no + 1
    batch_size = args.batch_size
    
    rewards = []
    avails = []
    
    print('Evaluation starts.')
        
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    for prob_no in range(start_no, end_no+1):
        fname = data_folder + '/%05d_env.pkl' % prob_no
        save_folder_prob = save_folder + '/%05d' % prob_no
        
        if not os.path.exists(save_folder_prob):
            os.makedirs(save_folder_prob)
        
        # load env from data folder
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        
        print('Evaluation on {}/{}.'.format(prob_no-start_no + 1, total_no))
        print('No. airliners: {}, helic: {}, crew: {}'.format(r.num_airliners,
                                                              r.num_helicopters,
                                                              r.num_crews))
        
        hourly_reward, hourly_a = run_test(fname, save_folder_prob, norm_r, 
                                           simulation_time, batch_size, interval,
                                           Rtype)
        
        rewards.append(hourly_reward)
        avails.append(hourly_a)
    
    rewards = np.array(rewards)
    avails = np.array(avails)
    
    print('Hourly Rewards')
    print('Mean: {}, Std: {}'.format(np.mean(rewards), np.std(rewards)))
    print('Plane Availability')
    print('Mean: {}, Std: {}'.format(np.mean(avails), np.std(avails)))
    print('Done.')
