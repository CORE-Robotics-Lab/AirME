# -*- coding: utf-8 -*-
"""
RL baseline scheduler
"""

# import random
import numpy as np
import torch
import torch.nn as nn
# import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import MultiStepLR

from basenets import DeepRMNet, DecimaNet
from utils import MD_node_helper

class DRMScheduler(object):
    """
    DeepRM Multi Decision Scheduler
        with batch support
        policy gradients w/ time-based baselines
    """    
    def __init__(self, max_planes, max_crews, in_dim, hid_dim, out_dim,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, milestones=[30, 80], lr_gamma=0.1):
        self.device = device
        
        self.max_planes = max_planes
        self.max_crews = max_crews
        
        self.model = DeepRMNet(in_dim, hid_dim, out_dim).to(self.device)

        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay    
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)

    def feat_helper(self, num_planes, planes, num_crews, crews,
                    num_planes_avail, num_crews_avail, 
                    crews_scheduled, planes_scheduled,
                    num_airliners):
        '''
        Generate input feature for DeepRMNet
            1. C x P image, pixel value = [info of crew-plane task]
            2. Plane other status = [separate tensor]
        '''
        task_dim = 5
        status_dim = 10 # 16 - 6
        
        # Batch_Size = 1
        # generate [B x Nc x Np x task_dim] image
        rel_tensor = np.zeros((1, self.max_crews, self.max_planes, task_dim), 
                              dtype=np.float32)
        
        # loop over active maintenance tasks
        for i in range(num_planes):
            if planes[i].task.crew > 0:
                # under repair / scheduled in previous time steps
                p_idx = planes[i].id - 1
                c_idx = planes[i].task.crew - 1
                
                rel_tensor[0, c_idx, p_idx, 0] = 1
                rel_tensor[0, c_idx, p_idx, 2] = planes[i].task.duration / 10.0
                rel_tensor[0, c_idx, p_idx, 3] = planes[i].task.cost / planes[i].task.duration
                rel_tensor[0, c_idx, p_idx, 4] = planes[i].task.cost / planes[i].task.duration
            elif planes[i].id in planes_scheduled:
                # Scheduled in current time step (so task cost and duration are N/A)
                p_idx = planes[i].id - 1
                c_idx = -1
                for j in range(len(planes_scheduled)):
                    if planes[i].id == planes_scheduled[j]:
                        c_idx = crews_scheduled[j] - 1
                
                assert c_idx >= 0
                
                rel_tensor[0, c_idx, p_idx, 1] = 1
        
        # generate plane other status [B x Np x status_dim]
        stat_tensor = np.zeros((1, self.max_planes, status_dim), dtype=np.float32)
        
        for i in range(num_planes):
            stat_tensor[0, i, 0] = planes[i].operating_hours / 10.0
            stat_tensor[0, i, 1] = planes[i].prob_use
            stat_tensor[0, i, 2] = planes[i].num_landings / 5.0
            
            if planes[i].is_broken:
                stat_tensor[0, i,3] = 1

            stat_tensor[0,i,4] = planes[i].flying_reward

            # mission features
            if planes[i].is_flying:
                # check if the plane has been scheduled in current time step
                if planes[i].id not in planes_scheduled:         
                    stat_tensor[0,i,5] = 1
                    stat_tensor[0,i,6] = planes[i].mission.duration / 10.0
                    stat_tensor[0,i,7] = planes[i].mission.progress / 10.0

            # [1 0] for airliners, [0 1] for helicopters
            if i < num_airliners:
                stat_tensor[0,i,8] = 1
            else:
                stat_tensor[0,i,9] = 1

        return rel_tensor, stat_tensor
    
    def new_feat_helper(self, repair):
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)        
        
        new_dict = {}
        new_dict['plane'] = np.zeros((1, self.max_planes+1, 16), dtype=np.float32)
        new_dict['plane'][0,:repair.num_planes+1] = feat_dict['plane']
        
        new_dict['crew'] = np.zeros((1, self.max_crews, 5))
        new_dict['crew'][0,:repair.num_crews] = feat_dict['crew']
        
        new_dict['state'] = feat_dict['state']
        
        return new_dict

    def get_multi_decisions(self, repair):
        # Both crews_avail and planes_avail gets updated every time a
        # (crew, plane) decision is generated
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0 or len(self.planes_avail) <= 0:
            return [], []
        
        # initialize decision queue / buffer
        # stores the decisions made so far in current time step
        self.plane_list = []
        self.crew_list = []
        
        # iterate over all available crews
        while self.crews_avail:
            # select a plane, this modifies self.planes_avail
            picked_plane = self.pick_plane(repair)
            # pick a crew
            picked_crew = self.crews_avail.pop()
        
            # update decision queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)
            
            # stop when 'void' action is selected
            if picked_plane == 0:
                break
            
        return self.crew_list.copy(), self.plane_list.copy()

    def pick_plane(self, repair):
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        feat_dict = self.new_feat_helper(repair)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)
            
        with torch.no_grad():
            result = self.model(feat_dict_tensor)
            q_logits = result.squeeze()
            
            # results output the raw logits for M+1 planes
            # plane 0, 1, 2, ..., M in the same order
            # need to convert to the valid set
            valid_plane = self.planes_avail.copy()
            valid_plane.append(0)            
            valid_q = q_logits[valid_plane]
            m = Categorical(logits = valid_q)
            a_idx = m.sample()
            idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no        

    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
    
    def batch_select_action(self, repair, i_b):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            return [], []
        
        # initialize decision queue / buffer
        # stores the decisions made so far in current time step
        self.plane_list = []
        self.crew_list = []
        # also records the log_prob of each decision in current time step
        self.tmp_log_prob = []
        
        # iterate over all available crews
        while self.crews_avail:
            # select a plane, this modifies self.planes_avail
            # also append selected log prob to tmp
            picked_plane = self.batch_pick_plane(repair)
            # pick a crew
            picked_crew = self.crews_avail.pop()
        
            # update decision queue
            # the graph will be built w.r.t. the updated queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)

            # stop when 'void' action is selected
            if picked_plane == 0:
                break          
            
        # append the sum of all log_probs in current time step
        self.batch_saved_log_probs[i_b].append(torch.sum(torch.stack(self.tmp_log_prob)))
            
        return self.crew_list.copy(), self.plane_list.copy()    

    def batch_pick_plane(self, repair):
        """
        Used in training / rollout collection
            i_b: the batch idx
        """         
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        feat_dict = self.new_feat_helper(repair)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)

        result = self.model(feat_dict_tensor)
        q_logits = result.squeeze()
        
        valid_plane = self.planes_avail.copy()
        valid_plane.append(0)            
        valid_q = q_logits[valid_plane]  
        m = Categorical(logits = valid_q)
        a_idx = m.sample()
        
        self.tmp_log_prob.append(m.log_prob(a_idx))
        
        idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no

    def batch_finish_episode(self, batch_size, sim_time, max_grad_norm):
        """
        Loss computation / Gradient updates
        """
        # 0. zero-pad episodes with early termination
        batch_returns = torch.zeros(batch_size, sim_time).to(self.device)
        
        # 1. compute total reward of each episode
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size] = self.batch_r(i_b)  
        
        # 2. compute time-based baseline values
        batch_baselines = torch.mean(batch_returns, dim=0)
        
        # 3. calculate advantages for each transition
        batch_advs = batch_returns - batch_baselines
        
        # 4. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        
        # 5. calculate loss for each episode in the batch
        # collect all log_pro into one tensor for computation
        # pad early termination with log_1 = 0.0
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # this gives a tensor of shape torch.Size([r_size])
            log_prob_tensor[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])
        
        # reset gradients
        self.optimizer.zero_grad()
        
        # sum up over all batches
        total_loss = torch.sum(-log_prob_tensor * batch_advs_norm) / batch_size
        loss_np = total_loss.data.cpu().numpy()
        
        # perform backprop
        total_loss.backward()
        
        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
        
        self.optimizer.step()
        
        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]

        return loss_np

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        R = 0.0
        returns = [] # list to save the true values

        for rw in self.batch_rewards[i_b][::-1]:
            # calculate the discounted value
            R = rw + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        return returns

    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()

class DecimaScheduler(object):
    """
    Decima Multi Decision Scheduler
        with batch support
        policy gradients w/ time-based baselines
        Use the same GNN formulation as described in Decima
    """    
    def __init__(self, in_dim, hid_dim, out_dim,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, milestones=[30, 80], lr_gamma=0.1):
        self.device = device
        
        self.model = DecimaNet(in_dim, hid_dim, out_dim).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay    
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)        
        
    def get_multi_decisions(self, repair):
        # Both crews_avail and planes_avail gets updated every time a
        # (crew, plane) decision is generated
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0 or len(self.planes_avail) <= 0:
            return [], []
        
        # initialize decision queue / buffer
        # stores the decisions made so far in current time step
        self.plane_list = []
        self.crew_list = []
        
        # iterate over all available crews
        while self.crews_avail:
            # select a plane, this modifies self.planes_avail
            picked_plane = self.pick_plane(repair)
            # pick a crew
            picked_crew = self.crews_avail.pop()
        
            # update decision queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)
            
        return self.crew_list.copy(), self.plane_list.copy()   
        
    def pick_plane(self, repair):        
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        feat_tensor = feat_dict['plane'][1:].copy()

        feat_t = torch.Tensor(feat_tensor).to(self.device)  

        with torch.no_grad():
            result = self.model(feat_t)
            q_logits = result.squeeze()  # q_logits is for all planes #1-#M
            
            valid_plane_idx = []
            for i in range(num_p_a):
                valid_plane_idx.append(self.planes_avail[i]-1)
                      
            valid_q = q_logits[valid_plane_idx]
            m = Categorical(logits = valid_q)
            a_idx = m.sample()
            idx = a_idx.item()

        # modify planes_avail
        plane_no = self.planes_avail.pop(idx)
        
        return plane_no        
        
    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]

    def batch_select_action(self, repair, i_b):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            return [], []
        
        # initialize decision queue / buffer
        # stores the decisions made so far in current time step
        self.plane_list = []
        self.crew_list = []
        # also records the log_prob of each decision in current time step
        self.tmp_log_prob = []
        
        # iterate over all available crews
        while self.crews_avail:
            # select a plane, this modifies self.planes_avail
            # also append selected log prob to tmp
            picked_plane = self.batch_pick_plane(repair)
            # pick a crew
            picked_crew = self.crews_avail.pop()
        
            # update decision queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)    
            
        # append the sum of all log_probs in current time step
        self.batch_saved_log_probs[i_b].append(torch.sum(torch.stack(self.tmp_log_prob)))
            
        return self.crew_list.copy(), self.plane_list.copy()        
        
    def batch_pick_plane(self, repair):
        """
        Used in training / rollout collection
            i_b: the batch idx
        """       
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        feat_tensor = feat_dict['plane'][1:].copy()

        feat_t = torch.Tensor(feat_tensor).to(self.device)  

        result = self.model(feat_t)
        q_logits = result.squeeze()
        
        valid_plane_idx = []
        for i in range(num_p_a):
            valid_plane_idx.append(self.planes_avail[i]-1)
            
        valid_q = q_logits[valid_plane_idx]  
        m = Categorical(logits = valid_q)
        a_idx = m.sample()
        
        self.tmp_log_prob.append(m.log_prob(a_idx))
        
        idx = a_idx.item()

        plane_no = self.planes_avail.pop(idx)
        
        return plane_no          

    def batch_finish_episode(self, batch_size, sim_time, max_grad_norm):
        """
        Loss computation / Gradient updates
        """
        # 0. zero-pad episodes with early termination
        batch_returns = torch.zeros(batch_size, sim_time).to(self.device)
        
        # 1. compute total reward of each episode
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size] = self.batch_r(i_b)  
        
        # 2. compute time-based baseline values
        batch_baselines = torch.mean(batch_returns, dim=0)
        
        # 3. calculate advantages for each transition
        batch_advs = batch_returns - batch_baselines
        
        # 4. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)
        
        # 5. calculate loss for each episode in the batch
        # collect all log_pro into one tensor for computation
        # pad early termination with log_1 = 0.0
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # this gives a tensor of shape torch.Size([r_size])
            log_prob_tensor[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])
        
        # reset gradients
        self.optimizer.zero_grad()
        
        # sum up over all batches
        total_loss = torch.sum(-log_prob_tensor * batch_advs_norm) / batch_size
        loss_np = total_loss.data.cpu().numpy()
        
        # perform backprop
        total_loss.backward()
        
        # perform Gradient Clipping-by-norm
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
        
        self.optimizer.step()
        
        # reset rewards and action buffer
        for i_b in range(batch_size):
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]

        return loss_np

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        R = 0.0
        returns = [] # list to save the true values

        for rw in self.batch_rewards[i_b][::-1]:
            # calculate the discounted value
            R = rw + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns).to(self.device)
        return returns

    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()
