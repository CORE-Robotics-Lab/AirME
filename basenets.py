# -*- coding: utf-8 -*-
"""
Deep learing baseline model
V1b - DeepRM
V2 - Decima
"""

import torch
import torch.nn as nn

class DeepRMNet(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim):
        super(DeepRMNet, self).__init__()
        
        self.layer1 = nn.Linear(in_dim, hid_dim)
        self.layer2 = nn.Linear(hid_dim, hid_dim)
        self.layer3 = nn.Linear(hid_dim, hid_dim)
        self.layer4 = nn.Linear(hid_dim, out_dim)
        
        self.relu = nn.ReLU()

    def forward(self, feat_dict):
        plane_f = torch.flatten(feat_dict['plane'], 1)
        crew_f = torch.flatten(feat_dict['crew'], 1)
        state_f = feat_dict['state']
        
        h0 = torch.cat([plane_f, crew_f, state_f], dim=1)
        
        h1 = self.relu(self.layer1(h0))
        h2 = self.relu(self.layer2(h1))
        h3 = self.relu(self.layer3(h2))
        h4 = self.layer4(h3)
                
        return h4

class DecimaNet(nn.Module):
    
    def __init__(self, in_dim, hid_dim, out_dim):
        '''
        use dict for dimensions
        '''
        super(DecimaNet, self).__init__()
        
        self.plane_layer1 = nn.Linear(in_dim['plane'], hid_dim['plane'])
        self.plane_layer2 = nn.Linear(hid_dim['plane'], hid_dim['plane'])
        self.plane_layer3 = nn.Linear(hid_dim['plane'], out_dim['plane'])

        assert out_dim['plane'] == in_dim['state']

        self.state_layer1 = nn.Linear(in_dim['state'], hid_dim['state'])
        self.state_layer2 = nn.Linear(hid_dim['state'], hid_dim['state'])
        self.state_layer3 = nn.Linear(hid_dim['state'], out_dim['state'])

        assert in_dim['q'] == out_dim['state'] + out_dim['plane']

        self.q_layer1 = nn.Linear(in_dim['q'], hid_dim['q'])
        self.q_layer2 = nn.Linear(hid_dim['q'], hid_dim['q'])
        self.q_layer3 = nn.Linear(hid_dim['q'], out_dim['q']) 

        self.relu = nn.ReLU()

    def forward(self, feat_tensor):
        # plane embeddings
        p1 = self.relu(self.plane_layer1(feat_tensor))
        p2 = self.relu(self.plane_layer2(p1))
        p3 = self.relu(self.plane_layer3(p2))   # M x Hp
                
        # global state embeddings
        s0 = torch.sum(p3, dim=0, keepdim=True)
        s1 = self.relu(self.state_layer1(s0))
        s2 = self.relu(self.state_layer2(s1))
        s3 = self.relu(self.state_layer3(s2))   # 1 x Hs
        
        # Q network part
        # predict q(state,plane) for all decisions/planes
        hs_list = []
        for i in range(feat_tensor.shape[0]):
            hs_list.append(s3)
        
        hs = torch.cat(hs_list, dim=0)      # M x Hs
        q0 = torch.cat([hs, p3], dim=1)     # M x (Hs+Hp)
        
        q1 = self.relu(self.q_layer1(q0))
        q2 = self.relu(self.q_layer2(q1))
        q3 = self.q_layer3(q2)      # M x 1
                
        return q3
