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
from heuristics import HybridScheduler, get_avail_with_prob
from heuristics import RandomScheduler

'''
Perform multiple runs using the same env
Inputs:
    fname: env file
    save_folder_prob: save folder for current env
    norm_r: boolen
    simulation_time: int
    batch_size: how many runs to perfrom
    h_t: heuristic type
        'random', 'simple', 'hybrid_hour', 'hybrid_landing', 'prob'
    hy_th: threshold value for hybrid heuristic
Return the mean values
'''
def run_test(fname, save_folder_prob, norm_r, simulation_time, batch_size, h_t, hy_th, Rtype):
    assert h_t in ['random', 'simple', 'hybrid_hour', 'hybrid_landing', 'prob']
    
    hourly_r_list = []
    hourly_a_list = []
    
    for i in range(batch_size):
        print('Batch {}/{}.'.format(i+1, batch_size))
        with open(fname, 'rb') as f:
            r = pickle.load(f)
        
        if h_t == 'random' or h_t == 'simple':
            scheduler = RandomScheduler(r.num_planes, r.num_crews)
        else:
            scheduler = HybridScheduler(r.num_planes, r.num_crews)
        
        time_t = []
        total_reward = []
        plane_avail = []        
        repair_cost = []
        flying_reward = []
        
        for t in range(simulation_time):
            if h_t == 'random':
                crews_avail, planes_avail = r.get_available()
                crew_list, plane_list = scheduler.get_multi_available_action(crews_avail, planes_avail)
            elif h_t == 'simple':
                crews_avail = r.get_available_crews()
                planes_broken, planes_avail = r.get_broken_and_available()                 
                crew_list, plane_list = scheduler.get_fix_broken(crews_avail, planes_broken)        
            elif h_t == 'hybrid_hour':
                crews_avail = r.get_available_crews()
                planes_broken, planes_avail = r.get_broken_and_available()
                crew_list, plane_list = scheduler.get_schedule_actions(crews_avail,
                                                                planes_broken,
                                                                planes_avail,
                                                                'hour', hy_th)
            elif h_t == 'hybrid_landing':
                crews_avail = r.get_available_crews()
                planes_broken, planes_avail = r.get_broken_and_available()
                crew_list, plane_list = scheduler.get_schedule_actions(crews_avail,
                                                                planes_broken,
                                                                planes_avail,
                                                                'landing', hy_th)
            elif h_t == 'prob':
                crews_avail = r.get_available_crews()
                planes_broken, _ = r.get_broken_and_available()
                planes_avail = get_avail_with_prob(r)
                crew_list, plane_list = scheduler.get_schedule_actions(crews_avail,
                                                                planes_broken,
                                                                planes_avail,
                                                                'prob', hy_th)
                
            cost, reward, avail_count = r.step_multi(crew_list, plane_list, 
                                                     verbose = False, norm_r = norm_r)
        
            time_t.append(r.total_hours)
            
            if Rtype == 'R1':                       
                act_reward = reward-cost
            elif Rtype == 'R2':
                act_reward = reward
            elif Rtype == 'R3':
                act_reward = avail_count / r.num_planes
            else:
                print('Error')            
            
            total_reward.append(act_reward)
            plane_avail.append(avail_count / r.num_planes)
            repair_cost.append(cost)
            flying_reward.append(reward)

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
        np.save(savefname_i+'_rcost', repair_cost)
        np.save(savefname_i+'_freward', flying_reward)
        
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
    parser.add_argument('--h-type', type=str)
    parser.add_argument('--hy-th', default=1, type=int)
    parser.add_argument('--Rtype', type=str)
    args = parser.parse_args()    
    
    simulation_time = args.sim_time

    print('Heurisitic Scheduler')

    norm_r = True

    h_type = args.h_type
    hy_th = args.hy_th
    token = '%04d' % hy_th
    
    if h_type == 'prob':
        hy_th = hy_th * 0.001
    
    print('Heurisitic type: ' + h_type)
    
    # Rtype: reward type
    Rtype = args.Rtype
    
    assert Rtype in ['R1', 'R2', 'R3']
    print(Rtype + ' reward')  
    
    data_folder = args.data_folder
    save_folder = args.save_folder + h_type + token
    
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
                                           simulation_time, batch_size, h_type,
                                           hy_th, Rtype)
        
        rewards.append(hourly_reward)
        avails.append(hourly_a)
    
    rewards = np.array(rewards)
    avails = np.array(avails)
    
    print('Hourly Rewards')
    print('Mean: {}, Std: {}'.format(np.mean(rewards), np.std(rewards)))
    print('Plane Availability')
    print('Mean: {}, Std: {}'.format(np.mean(avails), np.std(avails)))
    print('Done.')
    print(hy_th)
