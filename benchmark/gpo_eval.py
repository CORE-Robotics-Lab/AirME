# -*- coding: utf-8 -*-
"""
Evaluation code for GPOScheduler

Evaluated in multi-decision space
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sys
import pickle
import time

import torch

sys.path.append('../')

from gposcheduler import MDScheduler, SDScheduler, VoidScheduler

def run_test(fname, save_folder_prob, norm_r, simulation_time, batch_size, Rtype):
    hourly_r_list = []
    hourly_a_list = []
    record_time = []
    
    # scheduler already being loaded outside of this function
    for i in range(batch_size):
        print('Batch {}/{}.'.format(i+1, batch_size))
        start_t = time.time()
        with open(fname, 'rb') as f:
            r = pickle.load(f)
    
        time_t = []
        total_reward = []
        plane_avail = []
        repair_cost = []
        flying_reward = []
        
        for t in range(simulation_time):
            crew_list, plane_list = scheduler.get_multi_decisions(r)
            
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
            
            print('time: %d, repair cost: %f, flying reward: %f' %
                  (r.total_hours, cost, reward), end='\r')
        
        savefname_i = save_folder_prob + '/%03d' % i
        end_t = time.time()
        record_time.append([i+1, end_t - start_t])
        print('Computation time: {:.4f} s'.format(end_t - start_t))
        
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

    # save computation time for current problem
    record_time_np = np.array(record_time, dtype=np.float32)
    time_save_path = save_folder_prob + '/runtime'
    np.save(time_save_path, record_time_np)
       
    return np.mean(hourly_r_list), np.mean(hourly_a_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str)
    parser.add_argument('--data-folder', type=str)
    parser.add_argument('--save-folder', type=str)
    parser.add_argument('--start-no', default=1, type=int)
    parser.add_argument('--end-no', default=10, type=int)
    parser.add_argument('--sim-time', default=720, type=int)
    parser.add_argument('--batch-size', default=10, type=int)
    parser.add_argument('--Rtype', type=str)
    parser.add_argument('--Stype', type=str)
    args = parser.parse_args()
        
    simulation_time = args.sim_time

    print('Multi-Decsion Scheduler')
    
    norm_r = True
    
    # Stype: scheduler type
    Stype = args.Stype
    assert Stype in ['void', 'MD', 'SD']
    print('Scheduler type: ' + Stype)
    
    # Rtype: reward type
    Rtype = args.Rtype
    
    assert Rtype in ['R1', 'R2', 'R3']
    print(Rtype + ' reward')
    
    data_folder = args.data_folder
    save_folder = args.save_folder
    
    start_no = args.start_no
    end_no = args.end_no
    total_no = end_no - start_no + 1
    batch_size = args.batch_size
    
    rewards = []
    avails = []
    
    '''
    Load HetG model
    '''
    device = torch.device('cuda')
    
    if Stype == 'SD':
        in_dim = {'plane': 14,
                  'crew': 4,
                  'state': 5,
                  'value': 1
                  }
    else:
        in_dim = {'plane': 16,
                  'crew': 5,
                  'state': 5,
                  'value': 1
                  }
    
    hid_dim = {'plane': 32,
               'crew': 32,
               'state': 32,
               'value':32
               }
    
    out_dim = {'plane': 32,
              'crew': 32,
              'state': 32,
              'value': 1
              }
    
    cetypes = [('crew', 'repairing', 'plane'),
               ('plane', 'repaired_by', 'crew'),
               ('plane', 'p_in', 'state'),
               ('crew', 'c_in', 'state'),
               ('state', 's_in', 'state'),
               ('plane', 'p_to', 'value'),
               ('state', 's_to', 'value'),
               ('value', 'v_to', 'value')]
    
    num_heads = 4
    
    if Stype == 'void':
        scheduler = VoidScheduler(in_dim, hid_dim, out_dim, cetypes, num_heads, 
                                  device)
    elif Stype == 'MD':
        scheduler = MDScheduler(in_dim, hid_dim, out_dim, cetypes, num_heads, 
                                device)    
    elif Stype == 'SD':
        scheduler = SDScheduler(in_dim, hid_dim, out_dim, cetypes, num_heads, 
                                device)  
    
    trained_checkpoint = args.checkpoint
    cp = torch.load(trained_checkpoint)
    scheduler.model.load_state_dict(cp['policy_net_state_dict'])
    print('Loaded: '+trained_checkpoint)
    print('Data Folder: '+data_folder)
    print('Evaluation starts.')
    print('Save Folder: '+save_folder)

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
                                           simulation_time, batch_size, Rtype)
        
        rewards.append(hourly_reward)
        avails.append(hourly_a)
    
    print('Hourly Rewards')
    print('Mean: {}, Std: {}'.format(np.mean(rewards), np.std(rewards)))
    print('Plane Availability')
    print('Mean: {}, Std: {}'.format(np.mean(avails), np.std(avails)))
    print('Done.')
