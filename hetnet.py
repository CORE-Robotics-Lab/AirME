# -*- coding: utf-8 -*-
"""
Heterogeneous Graph Neural Network for rollout policcy

Modified for Aircraft Maintenance Scheduling
"""

import torch
import torch.nn as nn

from graph.hetgatv2 import MultiHeteroGATLayer

# Default GNN
class HetNet(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HetNet, self).__init__()
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')
    
    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        
        return h3

# GNN - 4 layer
class HetNet4Layer(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HetNet4Layer, self).__init__()
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer4 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')
    
    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        h4 = self.layer4(g, h3)
        
        return h4

# GNN - Actor Critic Verison
class HetNetAC(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HetNetAC, self).__init__()
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')
        
        self.critic_head = nn.Linear(out_dim['state'], 1)
        self.relu = nn.ReLU()
    
    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
        
        # 1x1
        critic_value = self.critic_head(self.relu(h3['state']))
        
        return h3, critic_value


# GNN - Actor Critic Verison
# Add time as a sperate input: denoted as key 'extra'
class HetNetACExtra(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, cetypes, num_heads=4):
        super(HetNetACExtra, self).__init__()
        
        hid_dim_input = {}
        for key in hid_dim:
            hid_dim_input[key] = hid_dim[key] * num_heads
        
        self.layer1 = MultiHeteroGATLayer(in_dim, hid_dim, cetypes, num_heads)
        self.layer2 = MultiHeteroGATLayer(hid_dim_input, hid_dim, cetypes, 
                                          num_heads)
        self.layer3 = MultiHeteroGATLayer(hid_dim_input, out_dim, cetypes, 
                                          num_heads, merge='avg')
        
        self.critic_head = nn.Linear(out_dim['extra'] + out_dim['state'], 1)
        
        self.extra_fc1 = nn.Linear(in_dim['extra'], out_dim['extra'])
        #self.extra_fc2 = nn.Linear(hid_dim['extra'], out_dim['extra'])
        self.relu = nn.ReLU()
        
    '''
    input
        g: DGL heterograph
            number of Q score nodes = number of available actions
        feat_dict: dictionary of input features
    '''
    def forward(self, g, feat_dict):
        # gnn
        h1 = self.layer1(g, feat_dict)
        h2 = self.layer2(g, h1)
        h3 = self.layer3(g, h2)
                
        # extra
        extra1 = self.relu(self.extra_fc1(feat_dict['extra']))
        #extra2 = self.relu(self.extra_fc2(extra1))
        
        # concat
        comb = torch.cat([self.relu(h3['state']), extra1], dim=1)
        
        # 1x1
        critic_value = self.critic_head(comb)
        
        return h3, critic_value
