# -*- coding: utf-8 -*-
"""
HetGPO for Aircraft Maintenance Scheduling
    with value function as baseline

    Train as a single decision policy
        R1 reward
    
    Ablation studies
"""

import os
import time
import random
import numpy as np
import pickle
import torch
import argparse

from utils import RepairEnv, get_default_param
from gposcheduler import ValueScheduler

parser = argparse.ArgumentParser()
parser.add_argument('--folder', type=str)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

# environment
random_init = True
percent_broken = 0.1
norm_r = True

# training parameters
resume_training = False
trained_checkpoint = './value01/checkpoint_00100.tar'

BATCH_SIZE = 8  # number of eposides/rollouts within a batch
GAMMA = 0.99

# episode length
ep_start = 50.0
ep_start_max = 200.0
ep_length = 40.0
ep_increase = 0.8

# learning rate
lr = 0.5*1e-3
weight_decay = 1e-3
milestones = [400, 1000]
lr_gamma = 0.5
max_grad_norm = 0.75

# random seed
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)

# PPO
n_updates_per_batch = 3
clip = 0.2
ent_coef = 0.01
target_kl = 0.02

total_batches = 3000
broken_thresh = 0.8

folder = args.folder

if not os.path.exists(folder):
    os.makedirs(folder)
    
device = torch.device('cuda')

loss_history = []
hourly_rs = []
hourly_std = []
max_hourly_r_per_crew = 0.0
max_hourly_r = 0.0
sim_time_history = []
size_history = []

'''
Initialize policy net
'''
in_dim = {'plane': 14,
          'crew': 4,
          'state': 5,
          'value': 1,
          'extra': 1
          }

hid_dim = {'plane': 32,
           'crew': 32,
           'state': 32,
           'value':32,
           'extra': 8
           }

out_dim = {'plane': 32,
          'crew': 32,
          'state': 32,
          'value': 1,
          'extra': 8
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

# policy_net inside of GNNScheduler
scheduler = ValueScheduler(in_dim, hid_dim, out_dim, cetypes, num_heads, 
                           device, GAMMA, lr, weight_decay,
                           milestones=milestones, lr_gamma=lr_gamma,
                           clip=clip, ent_coef=ent_coef, target_kl=target_kl)

if resume_training:
    cp = torch.load(trained_checkpoint)
    scheduler.model.load_state_dict(cp['policy_net_state_dict'])
    scheduler.optimizer.load_state_dict(cp['optimizer_state_dict'])
    scheduler.lr_scheduler.load_state_dict(cp['scheduler_state_dict'])
    start_batch = cp['i_batch'] + 1
    loss_history = cp['loss']
    hourly_rs = cp['hourly_rs']
    hourly_std = cp['hourly_std']
    sim_time_history = cp['sim_time']
    size_history = cp['size']
    max_hourly_r = np.max(hourly_rs)
    crew_size = np.array(cp['size'])[:,2]
    hourly_rs_per_crew = np.array(hourly_rs) / crew_size
    max_hourly_r_per_crew = np.max(hourly_rs_per_crew)
    if ep_start + ep_increase * (start_batch - 1) < ep_start_max:
        ep_start = ep_start + ep_increase * (start_batch - 1)
    else:
        ep_start = ep_start_max
    print(ep_start)
    print(max_hourly_r)
    print(max_hourly_r_per_crew)
    print('Checkpoint loaded.')
else:
    start_batch = 1  

# initialize batch buffer
scheduler.initialize_batch(BATCH_SIZE)

print('Initialization done')

'''
Main training loop
'''
for i_batch in range(start_batch, total_batches+1):
    '''
    Initialize
        Episodes within a batch use the same length/simulation_time
    '''
    start_t = time.time()
    batch_hourly_rs = []

    simulation_time = round(random.uniform(ep_start, ep_start + ep_length))

    if ep_start < ep_start_max:
        ep_start += ep_increase

    print('Training batch: {:d}, total length: {:d}'.format(i_batch, simulation_time))
    
    '''
    Use the same initialized env for the batch
    '''
    num_airliners = random.randint(16, 24)
    num_helicopters = random.randint(8, 12)
    num_crews = random.randint(6, 8)    
    
    print('No. airliners: {}, helic: {}, crew: {}'.format(num_airliners,
          num_helicopters, num_crews))    
    
    plane_info = get_default_param(num_airliners, num_helicopters, random_init,
                                   percent_broken)    
    
    # num_planes = num_airliners + num_helicopters
    
    r = RepairEnv(num_airliners, num_helicopters,
                  plane_info['num_parts_list'], plane_info['scale_lists'], 
                  plane_info['shape_lists'], plane_info['prob_uses'], plane_info['hour_scales'],
                  plane_info['hours_list'], plane_info['num_landings_list'],
                  plane_info['broken_list'], plane_info['reward_list'],
                  num_crews)    
    
    buffer_name = folder+'/buffer_env.pkl'
    with open(buffer_name, 'wb') as f:
        pickle.dump(r, f)
    
    '''
    Run multiple episodes
    '''
    for i_b in range(BATCH_SIZE):
        # load environment
        if i_b > 0:
            with open(buffer_name, 'rb') as f:
                r = pickle.load(f)
        
        for t in range(simulation_time):
            crew_no, plane_no = scheduler.batch_select_action(r, i_b,
                                                              extra = simulation_time - t)
    
            cost, reward, avail_count = r.step(crew_no, plane_no, verbose = False, 
                                               norm_r = norm_r, fix_break = False)
                        
            scheduler.batch_rewards[i_b].append(reward-cost)
            
            print('time: {:d}, reward: {:.4f}'.format(r.total_hours, reward-cost), end='\r')

            # if broken > threshhold, then terminate early
            broken_per = r.get_broken_rate()
            if broken_per >= broken_thresh:
                print('Early termination as broken rate hits {:.4f}'.
                      format(broken_per))
                break
    
        batch_hourly_rs.append(np.sum(scheduler.batch_rewards[i_b])/simulation_time)
    
    normalized_reward = batch_hourly_rs
    
    hourly_rs.append(np.mean(normalized_reward))
    hourly_std.append(np.std(normalized_reward))
    sim_time_history.append(simulation_time)
    size_history.append([num_airliners, num_helicopters, num_crews])

    # save this model if its hourly_r_per_crew is the highest so far
    if hourly_rs[-1] / r.num_crews > max_hourly_r_per_crew:
        max_hourly_r_per_crew = np.mean(normalized_reward) / r.num_crews
        max_save_path = folder+'/max_checkpoint_{:05d}.tar'.format(i_batch)
        torch.save({
            'i_batch': i_batch,
            'policy_net_state_dict': scheduler.model.state_dict(),
            'optimizer_state_dict': scheduler.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.lr_scheduler.state_dict(),
            'loss': loss_history,
            'hourly_rs': hourly_rs,
            'hourly_std': hourly_std,
            'sim_time': sim_time_history,
            'size': size_history
        }, max_save_path)
        print('Max model saved                   .')

    # save this model if its hourly_r is the highest so far
    if hourly_rs[-1] > max_hourly_r:
        max_hourly_r = np.mean(normalized_reward)
        max_save_path = folder+'/hr_checkpoint_{:05d}.tar'.format(i_batch)
        torch.save({
            'i_batch': i_batch,
            'policy_net_state_dict': scheduler.model.state_dict(),
            'optimizer_state_dict': scheduler.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.lr_scheduler.state_dict(),
            'loss': loss_history,
            'hourly_rs': hourly_rs,
            'hourly_std': hourly_std,
            'sim_time': sim_time_history,
            'size': size_history
        }, max_save_path)
        print('Hr model saved                    .')

    '''
    Perform training when all batch episodes finish
    '''
    if i_batch > 1:
        scheduler.adjust_lr()
        
    loss = scheduler.batch_finish_episode(BATCH_SIZE, simulation_time, 
                                          max_grad_norm, n_updates_per_batch)
    loss_history.append(loss['total'])

    end_t = time.time()
    print('[Batch {}], loss: {:e}, hourly reward: {:.4f}, time: {:.3f} s'.
          format(i_batch, loss_history[-1], hourly_rs[-1], end_t - start_t))

    '''
    Save checkpoints
    '''
    if i_batch % 10 == 0:
        checkpoint_path = folder+'/checkpoint_{:05d}.tar'.format(i_batch)
        torch.save({
            'i_batch': i_batch,
            'policy_net_state_dict': scheduler.model.state_dict(),
            'optimizer_state_dict': scheduler.optimizer.state_dict(),
            'scheduler_state_dict': scheduler.lr_scheduler.state_dict(),
            'loss': loss_history,
            'hourly_rs': hourly_rs,
            'hourly_std': hourly_std,
            'sim_time': sim_time_history,
            'size': size_history
        }, checkpoint_path)
        print('checkpoint saved to '+checkpoint_path)

print('Complete')
