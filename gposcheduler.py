# -*- coding: utf-8 -*-
"""
HetGPO Scheduler class
    Graph-based PPO learner
"""

import random
import numpy as np
import scipy.signal
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim.lr_scheduler import MultiStepLR

from hetnet import HetNet, HetNetACExtra
from utils import hetgraph_node_helper, build_hetgraph, MD_build_hetgraph, MD_node_helper


# https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/ppo/core.py
def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input: 
        vector x, 
        [x0, 
         x1, 
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MDScheduler(object):
    """
    [New] GPO Multi Decision Scheduler
        with batch support
        use time-based baselines
        train with multi-decision in the loop
        clip: Clipping parameter
        ent_coef: Entropy coefficient for the loss calculation
    """
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, milestones=[30, 80], lr_gamma=0.1,
                 clip=0.2, ent_coef=0.01, target_kl=None):
        self.device = device
        
        self.model = HetNet(in_dim, hid_dim, out_dim, cetypes,
                            num_heads).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay    
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)

        self.clip = clip
        self.ent_coef = ent_coef
        self.target_kl = target_kl

    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
        
        # Add buffers for the information needed to recompute the log_prob of new policy
        # fdt stores feat_dict (np array)
        self.batch_fd = [[] for i in range(batch_size)]
        # hetg stores hetg (on CPU)
        self.batch_hetg = [[] for i in range(batch_size)]        
        # act stores decisons {plane_list + crew_list}
        self.batch_act_plane = [[] for i in range(batch_size)]
        self.batch_act_crew = [[] for i in range(batch_size)]
        # a_idx stores decision idx in Categorial
        self.batch_a_idx = [[] for i in range(batch_size)]
        
    def batch_select_action(self, repair, i_b):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        self.batch_fd[i_b].append([]) # this list can be accessed by self.batch_fd[i_b][-1]
        self.batch_hetg[i_b].append([])
        self.batch_a_idx[i_b].append([])

        if len(self.crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            self.batch_act_plane[i_b].append([])
            self.batch_act_crew[i_b].append([])
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
            picked_plane = self.batch_pick_plane(repair, i_b)
            # pick a crew
            picked_crew = self.crews_avail.pop()
        
            # update decision queue
            # the graph will be built w.r.t. the updated queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)
            
        # append the sum of all log_probs in current time step
        self.batch_saved_log_probs[i_b].append(torch.sum(torch.stack(self.tmp_log_prob)))
        
        # append the decision list
        self.batch_act_plane[i_b].append(copy.deepcopy(self.plane_list))
        self.batch_act_crew[i_b].append(copy.deepcopy(self.crew_list))
            
        return self.crew_list.copy(), self.plane_list.copy()

    def batch_pick_plane(self, repair, i_b):
        """
        Used in training / rollout collection
            i_b: the batch idx
        """      
        # build graph
        hetg = MD_build_hetgraph(repair.num_planes, repair.num_crews, 
                                 repair.get_crew_plane_list(),
                                 self.planes_avail, self.crew_list, self.plane_list,
                                 torch.device('cpu'))      
        
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        # store fd and hetg
        self.batch_fd[i_b][-1].append(copy.deepcopy(feat_dict))
        self.batch_hetg[i_b][-1].append(copy.deepcopy(hetg))
        
        hetg = hetg.to(self.device)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)  

        with torch.no_grad():
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = m.sample()
            
            # store a_idx
            self.batch_a_idx[i_b][-1].append(copy.deepcopy(a_idx))
        
            # this appends a scalar (zero-dimensional tensor)
            self.tmp_log_prob.append(m.log_prob(a_idx))
            
            idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no

    def batch_finish_episode(self, batch_size, sim_time,
                             max_grad_norm,
                             n_updates_per_batch):
        """
        Loss computation / Gradient updates
            n_updates_per_batch: number of gradient updates per rollout collection 
        """
        '''
        4.1 Compute advantage
        '''
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
        
        '''
        For num_updates_per_batch {
            // Update policy with PPO-Clip objective
            // This is done in a for loop instead of just once
        '''
        for i in range(n_updates_per_batch):
            print('Updates {}/{}...'.format(i+1, n_updates_per_batch))
        	# 4.2 Recompute the log_pro using new policy
            # log_prob_new and log_prob_old has the same shape [Batch Size] x [Sim Time]
            log_prob_new, entropy, num_valid = self.recompute(batch_size, sim_time)
            
            # collect all log_prob_old into one tensor for computation
            # pad early termination with log_1 = 0.0
            log_prob_old = torch.zeros(batch_size, sim_time).to(self.device)
            for i_b in range(batch_size):
                r_size = len(self.batch_rewards[i_b])
                # this gives a tensor of shape torch.Size([r_size])
                log_prob_old[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])            
                
            # 4.3 compute the PPO-clip objective for loss
            # calculate ratios
            ratios = torch.exp(log_prob_new - log_prob_old)
            # calculate surrogate losses
            surr1 = ratios * batch_advs_norm
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advs_norm
            # sum up all time steps / num_valid
            total_loss = torch.sum(-torch.min(surr1, surr2)) / num_valid - entropy * self.ent_coef
            # mean = sum / (batch_size * sim_time)
            # total_loss = (-torch.min(surr1, surr2)).mean() - entropy * self.ent_coef
            loss_np = total_loss.data.cpu().numpy()
            
            '''
            Case 1 When log_prob_old = 0 in [:r_size]
                it means there is no crew available
                batch_saved_log_probs uses 0.0 -> log_prob_old = 0
                so we need set log_prob_new = 0 [no gradients]
                then ratio = e^0 = 1
                min(surr1, surr2) = advs
                but this is fine because there is not gradient for this

            Case 2 When early termination happens
                it means the returns = 0 for those left time steps
                but adv may not be zero
                log_prob_old = 0 for those left time steps
                so we also need set log_prob_new = 0 [no gradients]
                it becomes the same as case 1
            '''
            # Calculate approximate form of reverse KL Divergence for early stopping
            # from stable baselines 3
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
            with torch.no_grad():
                log_ratio = log_prob_new - log_prob_old
                approx_kl_div = torch.sum((torch.exp(log_ratio) - 1) - log_ratio) / num_valid
                approx_kl_div = approx_kl_div.cpu().numpy()
            
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"Early stopping at step {i+1} due to reaching max kl: {approx_kl_div:.2f}")
                break            
            
            # reset gradients
            self.optimizer.zero_grad()
            
            # perform backprop
            total_loss.backward()
            
            # perform Gradient Clipping-by-norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
            
            self.optimizer.step()
            
        # reset rewards and action buffer
        # use nested for loop
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
                        
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]
            
            for st in range(r_size):
                del self.batch_fd[i_b][st][:]
                del self.batch_hetg[i_b][st][:]
                del self.batch_act_plane[i_b][st][:]
                del self.batch_act_crew[i_b][st][:]
                del self.batch_a_idx[i_b][st][:]
            
            del self.batch_fd[i_b][:]
            del self.batch_hetg[i_b][:]
            del self.batch_act_plane[i_b][:]
            del self.batch_act_crew[i_b][:]
            del self.batch_a_idx[i_b][:]
        
        torch.cuda.empty_cache()
        self.initialize_batch(batch_size)
        
        return loss_np

    def recompute(self, batch_size, sim_time):
        """
        Re-compute the log_prob from saved rollout buffer
            1. when log_prob_old = 0 in [:r_size], set log_prob_new = 0
                -> pad with 0
            2. when early termination happens
                -> pad with 0
            Also compute the entropy
        """
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        entropy_list = [] # append entropy of all valid time steps from batch_size * sim_time
        
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # compute log_prob, entropy of each time step
            for st in range(r_size):
                new_log_prob, valid, new_entropy = self.compute_one(i_b, st)
                if valid:
                    log_prob_tensor[i_b][st] = new_log_prob
                    entropy_list.append(new_entropy)
        
        # this tells how many valid time steps are in the batch buffer
        num_valid = len(entropy_list)
        entropy = torch.mean(torch.stack(entropy_list))
        
        return log_prob_tensor, entropy, num_valid
    
    def compute_one(self, i_b, st):
        """
        Compute the log_prob at a specific time step in rollout buffer
            i_b: batch position
            st: time position
            Also compute the entropy = mean over all decisions' entropies
        """
        # plane_list = self.batch_act_plane[i_b][st]
        cr_list = self.batch_act_crew[i_b][st]
        tmp_log_prob = []
        tmp_entropy = []
        
        # when no crew is available, return False -> no valid time step
        if len(cr_list) <= 0:
            return 0.0, False, None
        
        # iterate over all decisions
        for j in range(len(cr_list)):
            hetg = self.batch_hetg[i_b][st][j]
            feat_dict = self.batch_fd[i_b][st][j]
            
            hetg = hetg.to(self.device)
            
            feat_dict_tensor = {}
            for key in feat_dict:
                feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)            
            
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = self.batch_a_idx[i_b][st][j]
            
            tmp_log_prob.append(m.log_prob(a_idx))
            tmp_entropy.append(m.entropy())
        
        return torch.sum(torch.stack(tmp_log_prob)), True, torch.mean(torch.stack(tmp_entropy))

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        # R = 0.0
        # returns = [] # list to save the true values

        # for rw in self.batch_rewards[i_b][::-1]:
        #     # calculate the discounted value
        #     R = rw + self.gamma * R
        #     returns.insert(0, R)
        
        # returns = torch.tensor(returns).to(self.device)
        # return returns
        rtg = discount_cumsum(self.batch_rewards[i_b], self.gamma)
        # ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array
        # with array.copy().) -> fixed
        rtgs = torch.Tensor(rtg.copy()).to(self.device)
        return rtgs

    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()

    def get_multi_decisions(self, repair):
        """
        Evaluation
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0:
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
            # the graph will be built w.r.t. the updated queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)
            
        return self.crew_list.copy(), self.plane_list.copy()        

    def pick_plane(self, repair):
        """
        Evaluation
        """      
        # build graph
        hetg = MD_build_hetgraph(repair.num_planes, repair.num_crews, 
                                 repair.get_crew_plane_list(),
                                 self.planes_avail, self.crew_list, self.plane_list,
                                 self.device)      
        
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)  

        with torch.no_grad():
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = m.sample()            
            idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no        

class SDScheduler(object):
    """
    [New] GPO (Single Decision) Scheduler
        with batch support
        use time-based baselines
        train as a single decision policy
        clip: Clipping parameter
        ent_coef: Entropy coefficient for the loss calculation
    """
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, milestones=[30, 80], lr_gamma=0.1,
                 clip=0.2, ent_coef=0.01, target_kl=None):
        self.device = device
        
        self.model = HetNet(in_dim, hid_dim, out_dim, cetypes,
                            num_heads).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay    
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)

        self.clip = clip
        self.ent_coef = ent_coef
        self.target_kl = target_kl

    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
        
        # Add buffers for the information needed to recompute the log_prob of new policy
        # fdt stores feat_dict (np array)
        self.batch_fd = [[] for i in range(batch_size)]
        # hetg stores hetg (on CPU)
        self.batch_hetg = [[] for i in range(batch_size)]
        # a_idx stores decision idx in Categorial
        self.batch_a_idx = [[] for i in range(batch_size)]
        
    def batch_select_action(self, repair, i_b):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        crews_avail, planes_avail = repair.get_available()

        if len(crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            self.batch_fd[i_b].append(None)
            self.batch_hetg[i_b].append(None)
            self.batch_a_idx[i_b].append(None)
            return 0, 0
        else:
            crew_no = random.choice(crews_avail)
        
        # get logits from GNN
        hetg = build_hetgraph(repair.num_planes, repair.num_crews, 
                              repair.get_crew_plane_list(),
                              planes_avail, torch.device('cpu'))   

        num_p_a = len(planes_avail)
        feat_dict = hetgraph_node_helper(repair.num_planes, repair.planes,
                                         repair.num_crews, repair.crews,
                                         num_p_a, len(crews_avail),
                                         repair.num_airliners,
                                         repair.total_fr)

        # store fd and hetg
        self.batch_fd[i_b].append(copy.deepcopy(feat_dict))
        self.batch_hetg[i_b].append(copy.deepcopy(hetg))

        hetg = hetg.to(self.device)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)

        with torch.no_grad():
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = m.sample()
            
            # store a_idx
            self.batch_a_idx[i_b].append(copy.deepcopy(a_idx))
            
            # this appends a scalar (zero-dimensional tensor)
            self.batch_saved_log_probs[i_b].append(m.log_prob(a_idx))
    
            idx = a_idx.item()
        
        if idx == num_p_a:
            # picked NULL
            plane_no = 0
        else:
            plane_no = planes_avail[idx]
        
        return crew_no, plane_no

    def batch_finish_episode(self, batch_size, sim_time,
                             max_grad_norm,
                             n_updates_per_batch):
        """
        Loss computation / Gradient updates
            n_updates_per_batch: number of gradient updates per rollout collection 
        """
        '''
        4.1 Compute advantage
        '''
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
        
        '''
        For num_updates_per_batch {
            // Update policy with PPO-Clip objective
            // This is done in a for loop instead of just once
        '''
        for i in range(n_updates_per_batch):
            print('Updates {}/{}...'.format(i+1, n_updates_per_batch))
        	# 4.2 Recompute the log_pro using new policy
            # log_prob_new and log_prob_old has the same shape [Batch Size] x [Sim Time]
            log_prob_new, entropy, num_valid = self.recompute(batch_size, sim_time)
            
            # collect all log_prob_old into one tensor for computation
            # pad early termination with log_1 = 0.0
            log_prob_old = torch.zeros(batch_size, sim_time).to(self.device)
            for i_b in range(batch_size):
                r_size = len(self.batch_rewards[i_b])
                # this gives a tensor of shape torch.Size([r_size])
                log_prob_old[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])            
                
            # 4.3 compute the PPO-clip objective for loss
            # calculate ratios
            ratios = torch.exp(log_prob_new - log_prob_old)
            # calculate surrogate losses
            surr1 = ratios * batch_advs_norm
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advs_norm
            # sum up all time steps / num_valid
            total_loss = torch.sum(-torch.min(surr1, surr2)) / num_valid - entropy * self.ent_coef
            # mean = sum / (batch_size * sim_time)
            # total_loss = (-torch.min(surr1, surr2)).mean() - entropy * self.ent_coef
            loss_np = total_loss.data.cpu().numpy()
            
            '''
            Case 1 When log_prob_old = 0 in [:r_size]
                it means there is no crew available
                batch_saved_log_probs uses 0.0 -> log_prob_old = 0
                so we need set log_prob_new = 0 [no gradients]
                then ratio = e^0 = 1
                min(surr1, surr2) = advs
                but this is fine because there is not gradient for this

            Case 2 When early termination happens
                it means the returns = 0 for those left time steps
                but adv may not be zero
                log_prob_old = 0 for those left time steps
                so we also need set log_prob_new = 0 [no gradients]
                it becomes the same as case 1
            '''
            # Calculate approximate form of reverse KL Divergence for early stopping
            # from stable baselines 3
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
            with torch.no_grad():
                log_ratio = log_prob_new - log_prob_old
                approx_kl_div = torch.sum((torch.exp(log_ratio) - 1) - log_ratio) / num_valid
                approx_kl_div = approx_kl_div.cpu().numpy()
            
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"Early stopping at step {i+1} due to reaching max kl: {approx_kl_div:.2f}")
                break            
            
            # reset gradients
            self.optimizer.zero_grad()
            
            # perform backprop
            total_loss.backward()
            
            # perform Gradient Clipping-by-norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
            
            self.optimizer.step()
            
        # reset rewards and action buffer
        # use nested for loop
        for i_b in range(batch_size):                        
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]
            del self.batch_fd[i_b][:]
            del self.batch_hetg[i_b][:]
            del self.batch_a_idx[i_b][:]
        
        torch.cuda.empty_cache()
        self.initialize_batch(batch_size)
        
        return loss_np

    def recompute(self, batch_size, sim_time):
        """
        Re-compute the log_prob from saved rollout buffer
            1. when log_prob_old = 0 in [:r_size], set log_prob_new = 0
                -> pad with 0
            2. when early termination happens
                -> pad with 0
            Also compute the entropy
        """
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        entropy_list = [] # append entropy of all valid time steps from batch_size * sim_time
        
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # compute log_prob, entropy of each time step
            for st in range(r_size):
                new_log_prob, valid, new_entropy = self.compute_one(i_b, st)
                if valid:
                    log_prob_tensor[i_b][st] = new_log_prob
                    entropy_list.append(new_entropy)
        
        # this tells how many valid time steps are in the batch buffer
        num_valid = len(entropy_list)
        entropy = torch.mean(torch.stack(entropy_list))
        
        return log_prob_tensor, entropy, num_valid
    
    def compute_one(self, i_b, st):
        """
        Compute the log_prob at a specific time step in rollout buffer
            i_b: batch position
            st: time position
            Also compute the entropy
        """
        a_idx = self.batch_a_idx[i_b][st]
        
        # when no crew is available, return False -> no valid time step
        if a_idx is None:
            return 0.0, False, None
        
        # do computation
        hetg = self.batch_hetg[i_b][st]
        feat_dict = self.batch_fd[i_b][st]
        
        hetg = hetg.to(self.device)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)            
        
        result = self.model(hetg, feat_dict_tensor)
        q_logits = result['value'].squeeze()
        m = Categorical(logits = q_logits)
                
        return m.log_prob(a_idx), True, m.entropy()

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        # R = 0.0
        # returns = [] # list to save the true values

        # for rw in self.batch_rewards[i_b][::-1]:
        #     # calculate the discounted value
        #     R = rw + self.gamma * R
        #     returns.insert(0, R)
        
        # returns = torch.tensor(returns).to(self.device)
        # return returns
        rtg = discount_cumsum(self.batch_rewards[i_b], self.gamma)
        # ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array
        # with array.copy().) -> fixed
        rtgs = torch.Tensor(rtg.copy()).to(self.device)
        return rtgs

    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()

    def get_multi_decisions(self, repair):
        """
        Evaluation
        """
        crews_avail, planes_avail = repair.get_available()

        if len(crews_avail) <= 0 or len(planes_avail) <= 0:
            return [], []
        
        crew_list = crews_avail.copy()
        plane_list = []

        # choose a plane / sample from softmax distribution
        # include null plane#0
        # build graph
        hetg = build_hetgraph(repair.num_planes, repair.num_crews, 
                              repair.get_crew_plane_list(),
                              planes_avail, self.device)         
        
        # generate feature tensor
        num_p_a = len(planes_avail)
        feat_dict = hetgraph_node_helper(repair.num_planes, repair.planes,
                                         repair.num_crews, repair.crews,
                                         num_p_a, len(crews_avail),
                                         repair.num_airliners,
                                         repair.total_fr)        
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)         
        
        with torch.no_grad():        
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            
            # pick a plane for every available crew
            for i in range(len(crews_avail)):
                m = Categorical(logits = q_logits)
                a_idx = m.sample()
                idx = a_idx.item()

                if idx == num_p_a:
                    plane_no = 0
                else:
                    plane_no = planes_avail[idx]
                    # modify logits if picked plane is not NULL
                    planes_avail.pop(idx)
                    num_p_a = len(planes_avail)
                    q_logits = torch.cat([q_logits[:idx], q_logits[idx+1:]])
                
                plane_list.append(plane_no)
        
        return crew_list, plane_list

class ValueScheduler(object):
    """
    [New] GPO (Single Decision) Scheduler
        with batch support
        use a learned value function
        train as a single decision policy
        clip: Clipping parameter
        ent_coef: Entropy coefficient for the loss calculation
    """
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, lmbda=0.95, milestones=[30, 80], lr_gamma=0.1,
                 clip=0.2, ent_coef=0.01, target_kl=None):
        self.device = device
        
        self.model = HetNetACExtra(in_dim, hid_dim, out_dim, cetypes,
                                   num_heads).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.weight_decay = weight_decay    
        self.saved_critics = []
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)

        self.clip = clip
        self.ent_coef = ent_coef
        self.target_kl = target_kl

    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
        self.batch_saved_critics = [[] for i in range(batch_size)]
        
        # Add buffers for the information needed to recompute the log_prob of new policy
        # fdt stores feat_dict (np array)
        self.batch_fd = [[] for i in range(batch_size)]
        # hetg stores hetg (on CPU)
        self.batch_hetg = [[] for i in range(batch_size)]
        # a_idx stores decision idx in Categorial
        self.batch_a_idx = [[] for i in range(batch_size)]
        
    def batch_select_action(self, repair, i_b, extra = 1):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        crews_avail, planes_avail = repair.get_available()

        if len(crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            crew_no = 0
        else:
            crew_no = random.choice(crews_avail)
        
        # get logits from GNN
        hetg = build_hetgraph(repair.num_planes, repair.num_crews, 
                              repair.get_crew_plane_list(),
                              planes_avail, torch.device('cpu'))   

        num_p_a = len(planes_avail)
        feat_dict = hetgraph_node_helper(repair.num_planes, repair.planes,
                                         repair.num_crews, repair.crews,
                                         num_p_a, len(crews_avail),
                                         repair.num_airliners,
                                         repair.total_fr)

        # add feature tensor for extra info
        feat_dict['extra'] = np.array([extra]).reshape(1,1)

        # store fd and hetg
        self.batch_fd[i_b].append(copy.deepcopy(feat_dict))
        self.batch_hetg[i_b].append(copy.deepcopy(hetg))

        hetg = hetg.to(self.device)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)

        with torch.no_grad():
            result, critic_value = self.model(hetg, feat_dict_tensor)
        
        if len(crews_avail) <= 0:
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            # store critic prediction when no action is available
            self.batch_saved_critics[i_b].append(critic_value)
            self.batch_a_idx[i_b].append(None)
            return crew_no, 0

        q_logits = result['value'].squeeze()
        m = Categorical(logits = q_logits)
        a_idx = m.sample()
        
        # store a_idx
        self.batch_a_idx[i_b].append(copy.deepcopy(a_idx))
        
        # this appends a scalar (zero-dimensional tensor)
        self.batch_saved_log_probs[i_b].append(m.log_prob(a_idx))
        self.batch_saved_critics[i_b].append(critic_value)

        idx = a_idx.item()        
        
        if idx == num_p_a:
            # picked NULL
            plane_no = 0
        else:
            plane_no = planes_avail[idx]
        
        return crew_no, plane_no

    def batch_finish_episode(self, batch_size, sim_time,
                             max_grad_norm,
                             n_updates_per_batch):
        """
        Loss computation / Gradient updates
            n_updates_per_batch: number of gradient updates per rollout collection 
        """
        '''
        4.1 Compute advantage
        '''
        # 0. zero-pad episodes with early termination
        batch_returns = torch.zeros(batch_size, sim_time).to(self.device)
        batch_advs = torch.zeros(batch_size, sim_time).to(self.device)
        
        # 1. compute total reward & advantage of each episode
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            batch_returns[i_b][:r_size], batch_advs[i_b][:r_size] = self.batch_GAE(i_b)
                
        # 2. normalize advantages within a batch
        eps = np.finfo(np.float32).eps.item()
        adv_mean = batch_advs.mean()
        adv_std = batch_advs.std()
        batch_advs_norm = (batch_advs - adv_mean) / (adv_std + eps)        
        
        '''
        For num_updates_per_batch {
            // Update policy with PPO-Clip objective
            // This is done in a for loop instead of just once
        '''
        for i in range(n_updates_per_batch):
            print('Updates {}/{}...'.format(i+1, n_updates_per_batch))
        	# 4.2 Recompute the log_pro using new policy
            # log_prob_new and log_prob_old has the same shape [Batch Size] x [Sim Time]
            log_prob_new, entropy, num_valid, value_preds = self.recompute(batch_size, sim_time)
            
            # collect all log_prob_old into one tensor for computation
            # pad early termination with log_1 = 0.0
            log_prob_old = torch.zeros(batch_size, sim_time).to(self.device)
            for i_b in range(batch_size):
                r_size = len(self.batch_rewards[i_b])
                # this gives a tensor of shape torch.Size([r_size])
                log_prob_old[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])            
                
            # 4.3 compute the PPO-clip objective for loss
            # calculate ratios
            ratios = torch.exp(log_prob_new - log_prob_old)
            # calculate surrogate losses
            surr1 = ratios * batch_advs_norm
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advs_norm
            # sum up all time steps / num_valid
            total_policy_loss = torch.sum(-torch.min(surr1, surr2)) / num_valid - entropy * self.ent_coef
            # mean = sum / (batch_size * sim_time)
            # total_loss = (-torch.min(surr1, surr2)).mean() - entropy * self.ent_coef
           
            total_critic_loss = F.mse_loss(value_preds, batch_returns)
            
            total_loss = total_policy_loss + total_critic_loss
           
            loss_np = {'total': total_loss.data.cpu().numpy(),
                       'policy': total_policy_loss.data.cpu().numpy(),
                       'critic': total_critic_loss.data.cpu().numpy()}
            
            '''
            Case 1 When log_prob_old = 0 in [:r_size]
                it means there is no crew available
                batch_saved_log_probs uses 0.0 -> log_prob_old = 0
                so we need set log_prob_new = 0 [no gradients]
                then ratio = e^0 = 1
                min(surr1, surr2) = advs
                but this is fine because there is not gradient for this

            Case 2 When early termination happens
                it means the returns = 0 for those left time steps
                but adv may not be zero
                log_prob_old = 0 for those left time steps
                so we also need set log_prob_new = 0 [no gradients]
                it becomes the same as case 1
            '''
            # Calculate approximate form of reverse KL Divergence for early stopping
            # from stable baselines 3
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
            with torch.no_grad():
                log_ratio = log_prob_new - log_prob_old
                approx_kl_div = torch.sum((torch.exp(log_ratio) - 1) - log_ratio) / num_valid
                approx_kl_div = approx_kl_div.cpu().numpy()
            
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"Early stopping at step {i+1} due to reaching max kl: {approx_kl_div:.2f}")
                break            
            
            # reset gradients
            self.optimizer.zero_grad()
            
            # perform backprop
            total_loss.backward()
            
            # perform Gradient Clipping-by-norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
            
            self.optimizer.step()
            
        # reset rewards and action buffer
        for i_b in range(batch_size):                        
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]
            del self.batch_fd[i_b][:]
            del self.batch_hetg[i_b][:]
            del self.batch_a_idx[i_b][:]
            del self.batch_saved_critics[i_b][:]
        
        torch.cuda.empty_cache()
        self.initialize_batch(batch_size)
        
        return loss_np

    def recompute(self, batch_size, sim_time):
        """
        Re-compute the log_prob from saved rollout buffer
            1. when log_prob_old = 0 in [:r_size], set log_prob_new = 0
                -> pad with 0
            2. when early termination happens
                -> pad with 0
            Also compute the entropy
        """
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        value_preds = torch.zeros(batch_size, sim_time).to(self.device)
        entropy_list = [] # append entropy of all valid time steps from batch_size * sim_time
        
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # compute log_prob, entropy, and value estimation of each time step
            for st in range(r_size):
                new_log_prob, valid, new_entropy, new_value = self.compute_one(i_b, st)
                value_preds[i_b][st] = new_value
                if valid:
                    log_prob_tensor[i_b][st] = new_log_prob
                    entropy_list.append(new_entropy)
        
        # this tells how many valid time steps are in the batch buffer
        num_valid = len(entropy_list)
        entropy = torch.mean(torch.stack(entropy_list))
        
        return log_prob_tensor, entropy, num_valid, value_preds
    
    def compute_one(self, i_b, st):
        """
        Compute the log_prob at a specific time step in rollout buffer
            i_b: batch position
            st: time position
            Also compute the entropy
        """
        a_idx = self.batch_a_idx[i_b][st]

        # do computation
        hetg = self.batch_hetg[i_b][st]
        feat_dict = self.batch_fd[i_b][st]
        
        hetg = hetg.to(self.device)
        
        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)            
        
        result, critic_value = self.model(hetg, feat_dict_tensor)
        
        # when no crew is available, return False -> no valid time step
        if a_idx is None:
            return 0.0, False, None, critic_value.squeeze()

        q_logits = result['value'].squeeze()
        m = Categorical(logits = q_logits)
                
        return m.log_prob(a_idx), True, m.entropy(), critic_value.squeeze()

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        # R = 0.0
        # returns = [] # list to save the true values

        # for rw in self.batch_rewards[i_b][::-1]:
        #     # calculate the discounted value
        #     R = rw + self.gamma * R
        #     returns.insert(0, R)
        
        # returns = torch.tensor(returns).to(self.device)
        # return returns
        rtg = discount_cumsum(self.batch_rewards[i_b], self.gamma)
        # ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array
        # with array.copy().) -> fixed
        rtgs = torch.Tensor(rtg.copy()).to(self.device)
        return rtgs

    '''
    Generalized Advantage Estimation
    '''
    def batch_GAE(self, i_b, Normalize = False):
        returns = []
        adv = []
        gae = 0.0
        
        for i in reversed(range(len(self.batch_rewards[i_b]))):
            if i == len(self.batch_rewards[i_b]) - 1:
                nextvalue = 0.0
                currentvalue = self.batch_saved_critics[i_b][i].item()
            else:
                nextvalue = self.batch_saved_critics[i_b][i+1].item()
                currentvalue = self.batch_saved_critics[i_b][i].item()
            
            delta = self.batch_rewards[i_b][i] + self.gamma * nextvalue - currentvalue
            gae = delta + self.gamma * self.lmbda * gae
            
            adv.insert(0, gae)
            returns.insert(0, gae + currentvalue)
        
        adv = torch.tensor(adv).to(self.device)
        returns = torch.tensor(returns).to(self.device)
        
        if Normalize:
            eps = np.finfo(np.float32).eps.item()
            adv = (adv - adv.mean()) / (adv.std() + eps)
        
        return returns, adv


    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()

# Skip Scheduler
class VoidScheduler(object):
    """
    [New] GPO Multi Decision (Void/Skip) Scheduler
        with batch support
        use time-based baselines
        train w/ void/skip token
        When scheduler picks void, decsion list for current time step concludes
            Indictating that the scheduler want to move to next time point
        clip: Clipping parameter
        ent_coef: Entropy coefficient for the loss calculation
    """
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads,
                 device=torch.device("cuda"), gamma=0.95, lr=1e-4,
                 weight_decay=1e-5, milestones=[30, 80], lr_gamma=0.1,
                 clip=0.2, ent_coef=0.01, target_kl=None):
        self.device = device
        
        self.model = HetNet(in_dim, hid_dim, out_dim, cetypes,
                            num_heads).to(self.device)
        
        self.saved_log_probs = []
        self.rewards = []
        self.gamma = gamma
        self.lr = lr
        self.weight_decay = weight_decay    
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr,
                                          weight_decay=self.weight_decay)

        self.lr_scheduler = MultiStepLR(self.optimizer, milestones = milestones,
                                        gamma = lr_gamma)

        self.clip = clip
        self.ent_coef = ent_coef
        self.target_kl = target_kl

    def initialize_batch(self, batch_size):
        """
        Initialize batch buffer
        """
        self.batch_saved_log_probs = [[] for i in range(batch_size)]
        self.batch_rewards = [[] for i in range(batch_size)]
        
        # Add buffers for the information needed to recompute the log_prob of new policy
        # fdt stores feat_dict (np array)
        self.batch_fd = [[] for i in range(batch_size)]
        # hetg stores hetg (on CPU)
        self.batch_hetg = [[] for i in range(batch_size)]        
        # act stores decisons {plane_list + crew_list}
        self.batch_act_plane = [[] for i in range(batch_size)]
        self.batch_act_crew = [[] for i in range(batch_size)]
        # a_idx stores decision idx in Categorial
        self.batch_a_idx = [[] for i in range(batch_size)]
        
    def batch_select_action(self, repair, i_b):
        """
        Rollout collection / Log prob saver / Multi-Decision in one action
            repair: the env
            i_b: the batch idx
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        self.batch_fd[i_b].append([]) # this list can be accessed by self.batch_fd[i_b][-1]
        self.batch_hetg[i_b].append([])
        self.batch_a_idx[i_b].append([])

        if len(self.crews_avail) <= 0:
            # append 0 when no crews are available
            # so P(pick #0) = 1, log_1 = 0
            self.batch_saved_log_probs[i_b].append(torch.tensor(0.0).to(self.device))
            self.batch_act_plane[i_b].append([])
            self.batch_act_crew[i_b].append([])
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
            picked_plane = self.batch_pick_plane(repair, i_b)
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
        
        # append the decision list
        self.batch_act_plane[i_b].append(copy.deepcopy(self.plane_list))
        self.batch_act_crew[i_b].append(copy.deepcopy(self.crew_list))
            
        return self.crew_list.copy(), self.plane_list.copy()

    def batch_pick_plane(self, repair, i_b):
        """
        Used in training / rollout collection
            i_b: the batch idx
        """      
        # build graph
        hetg = MD_build_hetgraph(repair.num_planes, repair.num_crews, 
                                 repair.get_crew_plane_list(),
                                 self.planes_avail, self.crew_list, self.plane_list,
                                 torch.device('cpu'))      
        
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        # store fd and hetg
        self.batch_fd[i_b][-1].append(copy.deepcopy(feat_dict))
        self.batch_hetg[i_b][-1].append(copy.deepcopy(hetg))
        
        hetg = hetg.to(self.device)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)  

        with torch.no_grad():
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = m.sample()
            
            # store a_idx
            self.batch_a_idx[i_b][-1].append(copy.deepcopy(a_idx))
        
            # this appends a scalar (zero-dimensional tensor)
            self.tmp_log_prob.append(m.log_prob(a_idx))
            
            idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no

    def batch_finish_episode(self, batch_size, sim_time,
                             max_grad_norm,
                             n_updates_per_batch):
        """
        Loss computation / Gradient updates
            n_updates_per_batch: number of gradient updates per rollout collection 
        """
        '''
        4.1 Compute advantage
        '''
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
        
        '''
        For num_updates_per_batch {
            // Update policy with PPO-Clip objective
            // This is done in a for loop instead of just once
        '''
        for i in range(n_updates_per_batch):
            print('Updates {}/{}...'.format(i+1, n_updates_per_batch))
        	# 4.2 Recompute the log_pro using new policy
            # log_prob_new and log_prob_old has the same shape [Batch Size] x [Sim Time]
            log_prob_new, entropy, num_valid = self.recompute(batch_size, sim_time)
            
            # collect all log_prob_old into one tensor for computation
            # pad early termination with log_1 = 0.0
            log_prob_old = torch.zeros(batch_size, sim_time).to(self.device)
            for i_b in range(batch_size):
                r_size = len(self.batch_rewards[i_b])
                # this gives a tensor of shape torch.Size([r_size])
                log_prob_old[i_b][:r_size] = torch.stack(self.batch_saved_log_probs[i_b])            
                
            # 4.3 compute the PPO-clip objective for loss
            # calculate ratios
            ratios = torch.exp(log_prob_new - log_prob_old)
            # calculate surrogate losses
            surr1 = ratios * batch_advs_norm
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * batch_advs_norm
            # sum up all time steps / num_valid
            total_loss = torch.sum(-torch.min(surr1, surr2)) / num_valid - entropy * self.ent_coef
            # mean = sum / (batch_size * sim_time)
            # total_loss = (-torch.min(surr1, surr2)).mean() - entropy * self.ent_coef
            loss_np = total_loss.data.cpu().numpy()
            
            '''
            Case 1 When log_prob_old = 0 in [:r_size]
                it means there is no crew available
                batch_saved_log_probs uses 0.0 -> log_prob_old = 0
                so we need set log_prob_new = 0 [no gradients]
                then ratio = e^0 = 1
                min(surr1, surr2) = advs
                but this is fine because there is not gradient for this

            Case 2 When early termination happens
                it means the returns = 0 for those left time steps
                but adv may not be zero
                log_prob_old = 0 for those left time steps
                so we also need set log_prob_new = 0 [no gradients]
                it becomes the same as case 1
            '''
            # Calculate approximate form of reverse KL Divergence for early stopping
            # from stable baselines 3
            # https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/ppo/ppo.py
            with torch.no_grad():
                log_ratio = log_prob_new - log_prob_old
                approx_kl_div = torch.sum((torch.exp(log_ratio) - 1) - log_ratio) / num_valid
                approx_kl_div = approx_kl_div.cpu().numpy()
            
            if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                print(f"Early stopping at step {i+1} due to reaching max kl: {approx_kl_div:.2f}")
                break            
            
            # reset gradients
            self.optimizer.zero_grad()
            
            # perform backprop
            total_loss.backward()
            
            # perform Gradient Clipping-by-norm
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = max_grad_norm)
            
            self.optimizer.step()
            
        # reset rewards and action buffer
        # use nested for loop
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
                        
            del self.batch_rewards[i_b][:]
            del self.batch_saved_log_probs[i_b][:]
            
            for st in range(r_size):
                del self.batch_fd[i_b][st][:]
                del self.batch_hetg[i_b][st][:]
                del self.batch_act_plane[i_b][st][:]
                del self.batch_act_crew[i_b][st][:]
                del self.batch_a_idx[i_b][st][:]
            
            del self.batch_fd[i_b][:]
            del self.batch_hetg[i_b][:]
            del self.batch_act_plane[i_b][:]
            del self.batch_act_crew[i_b][:]
            del self.batch_a_idx[i_b][:]
        
        torch.cuda.empty_cache()
        self.initialize_batch(batch_size)
        
        return loss_np

    def recompute(self, batch_size, sim_time):
        """
        Re-compute the log_prob from saved rollout buffer
            1. when log_prob_old = 0 in [:r_size], set log_prob_new = 0
                -> pad with 0
            2. when early termination happens
                -> pad with 0
            Also compute the entropy
        """
        log_prob_tensor = torch.zeros(batch_size, sim_time).to(self.device)
        entropy_list = [] # append entropy of all valid time steps from batch_size * sim_time
        
        for i_b in range(batch_size):
            r_size = len(self.batch_rewards[i_b])
            # compute log_prob, entropy of each time step
            for st in range(r_size):
                new_log_prob, valid, new_entropy = self.compute_one(i_b, st)
                if valid:
                    log_prob_tensor[i_b][st] = new_log_prob
                    entropy_list.append(new_entropy)
        
        # this tells how many valid time steps are in the batch buffer
        num_valid = len(entropy_list)
        entropy = torch.mean(torch.stack(entropy_list))
        
        return log_prob_tensor, entropy, num_valid
    
    def compute_one(self, i_b, st):
        """
        Compute the log_prob at a specific time step in rollout buffer
            i_b: batch position
            st: time position
            Also compute the entropy = mean over all decisions' entropies
        """
        # plane_list = self.batch_act_plane[i_b][st]
        cr_list = self.batch_act_crew[i_b][st]
        tmp_log_prob = []
        tmp_entropy = []
        
        # when no crew is available, return False -> no valid time step
        if len(cr_list) <= 0:
            return 0.0, False, None
        
        # iterate over all decisions
        for j in range(len(cr_list)):
            hetg = self.batch_hetg[i_b][st][j]
            feat_dict = self.batch_fd[i_b][st][j]
            
            hetg = hetg.to(self.device)
            
            feat_dict_tensor = {}
            for key in feat_dict:
                feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)            
            
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = self.batch_a_idx[i_b][st][j]
            
            tmp_log_prob.append(m.log_prob(a_idx))
            tmp_entropy.append(m.entropy())
        
        return torch.sum(torch.stack(tmp_log_prob)), True, torch.mean(torch.stack(tmp_entropy))

    def batch_r(self, i_b):
        """
        Compute total reward of each episode
        """
        # R = 0.0
        # returns = [] # list to save the true values

        # for rw in self.batch_rewards[i_b][::-1]:
        #     # calculate the discounted value
        #     R = rw + self.gamma * R
        #     returns.insert(0, R)
        
        # returns = torch.tensor(returns).to(self.device)
        # return returns
        rtg = discount_cumsum(self.batch_rewards[i_b], self.gamma)
        # ValueError: At least one stride in the given numpy array is negative, 
        # and tensors with negative strides are not currently supported. 
        # (You can probably work around this by making a copy of your array
        # with array.copy().) -> fixed
        rtgs = torch.Tensor(rtg.copy()).to(self.device)
        return rtgs

    def adjust_lr(self, metrics=0.0):
        """
        Adjust learning rate using lr_scheduler
        """
        self.lr_scheduler.step()

    def get_multi_decisions(self, repair):
        """
        Evaluation
        """
        self.crews_avail, self.planes_avail = repair.get_available()

        if len(self.crews_avail) <= 0:
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
            # the graph will be built w.r.t. the updated queue
            self.plane_list.append(picked_plane)
            self.crew_list.append(picked_crew)

            # stop when 'void' action is selected
            if picked_plane == 0:
                break
            
        return self.crew_list.copy(), self.plane_list.copy()        

    def pick_plane(self, repair):
        """
        Evaluation
        """      
        # build graph
        hetg = MD_build_hetgraph(repair.num_planes, repair.num_crews, 
                                 repair.get_crew_plane_list(),
                                 self.planes_avail, self.crew_list, self.plane_list,
                                 self.device)      
        
        # generate feature tensor
        num_p_a = len(self.planes_avail)
        num_c_a = len(self.crews_avail)
        feat_dict = MD_node_helper(repair.num_planes, repair.planes,
                                   repair.num_crews, repair.crews,
                                   num_p_a, num_c_a, self.crew_list, self.plane_list,
                                   repair.num_airliners, repair.total_fr)

        feat_dict_tensor = {}
        for key in feat_dict:
            feat_dict_tensor[key] = torch.Tensor(feat_dict[key]).to(self.device)  

        with torch.no_grad():
            result = self.model(hetg, feat_dict_tensor)
            q_logits = result['value'].squeeze()
            m = Categorical(logits = q_logits)
            a_idx = m.sample()            
            idx = a_idx.item()

        if idx == num_p_a:
            plane_no = 0
        else:
            # modify planes_avail if picked plane is not NULL(#0)
            plane_no = self.planes_avail.pop(idx)
        
        return plane_no
