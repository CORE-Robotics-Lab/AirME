# -*- coding: utf-8 -*-
"""
Policy gradient for Aircraft Maintenance Scheduling
    with time-based baseline

    Train under different objectives
    
RL baseline learner
"""

import os
import time
import random
import numpy as np
import pickle
import torch
import argparse

from utils import RepairEnv, get_default_param
from basescheduler import DRMScheduler, DecimaScheduler

parser = argparse.ArgumentParser()
# generel parameters
parser.add_argument('--folder', type=str)
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--Rtype', type=str)
parser.add_argument('--Stype', type=str)
# training parameters
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--lr_gamma', type=float, default=0.4)
parser.add_argument('--max_grad_norm', type=float, default=0.75)
parser.add_argument('--total_batches', type=int, default=3000)
# episode length paramters
parser.add_argument('--ep_start', type=float, default=50.0)
parser.add_argument('--ep_start_max', type=float, default=200.0)
parser.add_argument('--ep_length', type=float, default=30.0)
parser.add_argument('--ep_increase', type=float, default=0.8)
# resume training
parser.add_argument('--resume_training', default=False, action='store_true',
                    help='Whether to load previous checkpoint')
parser.add_argument('--trained_cp', type=str, default='tmp.tar')
# train on large or small
parser.add_argument('--scale', type=str, default='small')

args = parser.parse_args()

# environment
random_init = True
percent_broken = 0.1
norm_r = True

# training parameters
resume_training = args.resume_training
trained_checkpoint = args.trained_cp

# problem scale
scale = args.scale
assert scale in ['small', 'large', 'extra']
print('Problem scale: ' + scale)

BATCH_SIZE = args.batch_size  # number of eposides/rollouts within a batch
GAMMA = args.gamma

# episode length
ep_start = args.ep_start
ep_start_max = args.ep_start_max
ep_length = args.ep_length
ep_increase = args.ep_increase

# learning rate
lr = args.lr
weight_decay = args.weight_decay
milestones = [500]
lr_gamma = args.lr_gamma
max_grad_norm = args.max_grad_norm

# random seed
seed = args.seed
torch.manual_seed(seed)
random.seed(seed)

total_batches = args.total_batches
broken_thresh = 0.8

Stype = args.Stype # 'drm' or 'decima'
Rtype = args.Rtype
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
assert Stype in ['drm', 'decima']
print('Scheduler type: ' + Stype)

if Stype == 'drm':
    if scale == 'small':   
        max_planes = 36
        max_crews = 8
        hid_dim = 64
    elif scale == 'large':
        max_planes = 72
        max_crews = 16
        hid_dim = 128
    elif scale == 'extra':
        max_planes = 144
        max_crews = 32
        hid_dim = 128       
    in_dim = (max_planes+1) * 16 + max_crews * 5 + 5
    out_dim = max_planes + 1
elif Stype == 'decima':
    in_dim = {'plane': 16,
              'state': 32,
              'q': 64
              }
    
    hid_dim = {'plane': 64,
       'state': 64,
       'q':64
       }
    
    out_dim = {'plane': 32,
          'state': 32,
          'q': 1
          }

# policy_net inside of Scheduler
if Stype == 'drm':
    scheduler = DRMScheduler(max_planes, max_crews, in_dim, hid_dim, out_dim, 
                             device, GAMMA, lr, weight_decay,
                             milestones=milestones, lr_gamma=lr_gamma)    
elif Stype == 'decima':
    scheduler = DecimaScheduler(in_dim, hid_dim, out_dim, device,
                                GAMMA, lr, weight_decay,
                                milestones=milestones, lr_gamma=lr_gamma)
        
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

assert Rtype in ['R1', 'R2', 'R3']
print('Training under' + Rtype + ' reward')

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
    if scale == 'small':
        num_airliners = random.randint(16, 24)
        num_helicopters = random.randint(8, 12)
        num_crews = random.randint(6, 8)
    elif scale == 'large':
        num_airliners = random.randint(32, 48)
        num_helicopters = random.randint(16, 24)
        num_crews = random.randint(12, 16)
    elif scale == 'extra':
        num_airliners = random.randint(64, 96)
        num_helicopters = random.randint(32, 48)
        num_crews = random.randint(24, 32)     
    
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
            crew_list, plane_list = scheduler.batch_select_action(r, i_b)
    
            cost, reward, avail_count = r.step_multi(crew_list, plane_list,
                                                     verbose = False, norm_r = norm_r,
                                                     fix_break = False)            
            
            if Rtype == 'R1':                       
                act_reward = reward-cost
            elif Rtype == 'R2':
                act_reward = reward
            elif Rtype == 'R3':
                act_reward = avail_count / r.num_planes

            scheduler.batch_rewards[i_b].append(act_reward)
            print('time: {:d}, reward: {:.4f}'.format(r.total_hours, act_reward), end='\r')

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
        
    loss = scheduler.batch_finish_episode(BATCH_SIZE, simulation_time, max_grad_norm)
    loss_history.append(loss)

    end_t = time.time()
    print('[Batch {}], loss: {:e}, hourly reward: {:.4f}, time: {:.3f} s'.
          format(i_batch, loss_history[-1], hourly_rs[-1], end_t - start_t))

    '''
    Save checkpoints
    '''
    if i_batch % 20 == 0:
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
