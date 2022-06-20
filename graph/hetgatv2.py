# -*- coding: utf-8 -*-
"""
GAT for HetGraph
    ref: https://docs.dgl.ai/en/latest/tutorials/hetero/1_basics.html

    Modified for Aircraft Maintenance Scheduling
    Ultilize DGL's multi-head trick (from homograph)
"""

import torch
import torch.nn as nn
# import torch.nn.functional as F

import dgl.function as fn
from dgl.ops import edge_softmax

# in_dim: dict of input feature dimension for each node
# out_dim: dict of output feature dimension for each node
# cetypes: reutrn of G.canonical_etypes
class HeteroGATLayer(nn.Module):
    
    def __init__(self, in_dim, out_dim, cetypes, num_heads,
                 l_alpha = 0.2, use_relu = True):
        super(HeteroGATLayer, self).__init__()
        self._num_heads = num_heads
        self._in_dim = in_dim
        self._out_dim = out_dim   
        
        '''
        GNN part - all edge types, plus 'self-processing' for crew and plane
            <src, etype, dst>
            crew c_in state
            crew repairing plane
            plane p_in state
            plane p_to value
            plane repaired_by crew
            state s_in state
            state s_to value
            value v_to value
        '''
        self.fc = nn.ModuleDict({
            'crew': nn.Linear(in_dim['crew'], out_dim['crew'] * num_heads),
            'plane': nn.Linear(in_dim['plane'], out_dim['plane'] * num_heads),
            'c_in': nn.Linear(in_dim['crew'], out_dim['state'] * num_heads),
            'repairing': nn.Linear(in_dim['crew'], out_dim['plane'] * num_heads),
            'p_in': nn.Linear(in_dim['plane'], out_dim['state'] * num_heads),
            'p_to': nn.Linear(in_dim['plane'], out_dim['value'] * num_heads),
            'repaired_by': nn.Linear(in_dim['plane'], out_dim['crew'] * num_heads),
            's_in': nn.Linear(in_dim['state'], out_dim['state'] * num_heads),
            's_to': nn.Linear(in_dim['state'], out_dim['value'] * num_heads),
            'v_to': nn.Linear(in_dim['value'], out_dim['value'] * num_heads)
            })
            
        self.leaky_relu = nn.LeakyReLU(negative_slope = l_alpha)
        self.use_relu = use_relu
        if self.use_relu:
            self.relu = nn.ReLU()
        
        '''
        Attention part - split into src and dst
            crew c_in state
            plane p_in state
        '''
        self.c_in_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.c_in_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.p_in_src = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        self.p_in_dst = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_dim['state'])))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')

        # attention
        nn.init.xavier_normal_(self.c_in_src, gain=gain)
        nn.init.xavier_normal_(self.c_in_dst, gain=gain)
        nn.init.xavier_normal_(self.p_in_src, gain=gain)
        nn.init.xavier_normal_(self.p_in_dst, gain=gain)  

    def forward(self, g, feat_dict):
        '''
        Feature transform for crew and plane themselves
        '''
        # feature of crew
        Whcrew = self.fc['crew'](feat_dict['crew']).view(-1, self._num_heads, self._out_dim['crew'])
        g.nodes['crew'].data['Wh_crew'] = Whcrew
        
        # feature of plane
        Whplane = self.fc['plane'](feat_dict['plane']).view(-1, self._num_heads, self._out_dim['plane'])
        g.nodes['plane'].data['Wh_plane'] = Whplane        
        
        '''
        Feature transform for each edge/relation type
        '''
        # crew c_in state
        Whc_in = self.fc['c_in'](feat_dict['crew']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['crew'].data['Wh_c_in'] = Whc_in
        
        # crew repairing plane
        Whrepairing = self.fc['repairing'](feat_dict['crew']).view(-1, self._num_heads, self._out_dim['plane'])
        g.nodes['crew'].data['Wh_repairing'] = Whrepairing
        
        #　plane p_in state
        Whp_in = self.fc['p_in'](feat_dict['plane']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['plane'].data['Wh_p_in'] = Whp_in
        
        # plane p_to value
        Whp_to = self.fc['p_to'](feat_dict['plane']).view(-1, self._num_heads, self._out_dim['value'])
        g.nodes['plane'].data['Wh_p_to'] = Whp_to
        
        # plane repaired_by crew
        Whrepaired_by = self.fc['repaired_by'](feat_dict['plane']).view(-1, self._num_heads, self._out_dim['crew'])
        g.nodes['plane'].data['Wh_repaired_by'] = Whrepaired_by
        
        # state s_in state
        Whs_in = self.fc['s_in'](feat_dict['state']).view(-1, self._num_heads, self._out_dim['state'])
        g.nodes['state'].data['Wh_s_in'] = Whs_in
        
        # state s_to value
        Whs_to = self.fc['s_to'](feat_dict['state']).view(-1, self._num_heads, self._out_dim['value'])
        g.nodes['state'].data['Wh_s_to'] = Whs_to
        
        # value v_to value
        Whv_to = self.fc['v_to'](feat_dict['value']).view(-1, self._num_heads, self._out_dim['value'])
        g.nodes['value'].data['Wh_v_to'] = Whv_to
        
        # for srctype, etype, dsttype in g.canonical_etypes:
        #     Wh = self.fc[etype](feat_dict[srctype])
        #     g.nodes[srctype].data['Wh_%s' % etype] = Wh
        '''
        Attention computation on subgraphs
        '''
        # crew c_in state
        Attn_src_c_in = (Whc_in * self.c_in_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_c_in = (Whs_in * self.c_in_dst).sum(dim=-1).unsqueeze(-1)
        g['c_in'].srcdata.update({'Attn_src_c_in': Attn_src_c_in})
        g['c_in'].dstdata.update({'Attn_dst_c_in': Attn_dst_c_in})
        
        g['c_in'].apply_edges(fn.u_add_v('Attn_src_c_in', 'Attn_dst_c_in', 'e_c_in'))
        e_c_in = self.leaky_relu(g['c_in'].edata.pop('e_c_in'))
        
        g['c_in'].edata['a_c_in'] = edge_softmax(g['c_in'], e_c_in)
        
        g['c_in'].update_all(fn.u_mul_e('Wh_c_in', 'a_c_in', 'm_c_in'),
                             fn.sum('m_c_in', 'ft_c_in'))        
        
        # plane p_in state
        Attn_src_p_in = (Whp_in * self.p_in_src).sum(dim=-1).unsqueeze(-1)
        Attn_dst_p_in = (Whs_in * self.p_in_dst).sum(dim=-1).unsqueeze(-1)
        g['p_in'].srcdata.update({'Attn_src_p_in': Attn_src_p_in}) 
        g['p_in'].dstdata.update({'Attn_dst_p_in': Attn_dst_p_in}) 

        g['p_in'].apply_edges(fn.u_add_v('Attn_src_p_in', 'Attn_dst_p_in', 'e_p_in'))
        e_p_in = self.leaky_relu(g['p_in'].edata.pop('e_p_in'))

        g['p_in'].edata['a_p_in'] = edge_softmax(g['p_in'], e_p_in)

        g['p_in'].update_all(fn.u_mul_e('Wh_p_in', 'a_p_in', 'm_p_in'),
                             fn.sum('m_p_in', 'ft_p_in'))       
        
        '''
        Feature update on subgraphs with no attention
            If some of the nodes in the graph has no in-edges, 
            DGL does not invoke message and reduce functions for 
            these nodes and fill their aggregated messages with zero. 
            Users can control the filled values via set_n_initializer(). 
            DGL still invokes apply_node_func if provided.
        '''
        # crew repairing plane
        g['repairing'].update_all(fn.copy_src('Wh_repairing', 'z_repairing'),
                                  fn.sum('z_repairing', 'ft_reparing'))
        # plane p_to value
        g['p_to'].update_all(fn.copy_src('Wh_p_to', 'z_p_to'), fn.sum('z_p_to', 'ft_p_to'))
        # plane repaired_by crew
        g['repaired_by'].update_all(fn.copy_src('Wh_repaired_by', 'z_repaired_by'),
                                    fn.sum('z_repaired_by', 'ft_repaired_by'))
        # state s_in state
        g['s_in'].update_all(fn.copy_src('Wh_s_in', 'z_s_in'), fn.sum('z_s_in', 'ft_s_in'))
        # state s_to value
        g['s_to'].update_all(fn.copy_src('Wh_s_to', 'z_s_to'), fn.sum('z_s_to', 'ft_s_to'))
        # value v_to value        
        g['v_to'].update_all(fn.copy_src('Wh_v_to', 'z_v_to'), fn.sum('z_v_to', 'ft_v_to'))
        
        '''
        Combine features from subgraphs
            Sum up to hi'
            Need to check if subgraph is valid
        '''        
        # new feature of crew
        Whcrew_new = g.nodes['crew'].data['Wh_crew'].clone()
        if g['repaired_by'].number_of_edges() > 0:
            Whcrew_new += g.nodes['crew'].data['ft_repaired_by']
            
        g.nodes['crew'].data['h'] = Whcrew_new
        
        # new feature of plane
        Whplane_new = g.nodes['plane'].data['Wh_plane'].clone()
        if g['repairing'].number_of_edges() > 0:
            Whplane_new += g.nodes['plane'].data['ft_reparing']
        
        g.nodes['plane'].data['h'] = Whplane_new
        
        # new feature of state
        Whstate_new = g.nodes['state'].data['Wh_s_in'] + \
            g.nodes['state'].data['ft_c_in'] + \
                g.nodes['state'].data['ft_p_in']
        
        g.nodes['state'].data['h'] = Whstate_new
        
        # new feature of value
        Whvalue_new = g.nodes['value'].data['Wh_v_to'] + \
            g.nodes['value'].data['ft_p_to'] + \
                g.nodes['value'].data['ft_s_to']

        g.nodes['value'].data['h'] = Whvalue_new
        
        # deal with relu activation
        if self.use_relu:
            return {ntype : self.relu(g.nodes[ntype].data['h']) for ntype in g.ntypes}
        else:
            return {ntype : g.nodes[ntype].data['h'] for ntype in g.ntypes}

# input similar to HeteroGATLayer
# merge = 'cat' or 'avg'
class MultiHeteroGATLayer(nn.Module):
    def __init__(self, in_dim, out_dim, cetypes, num_heads, merge='cat'):
        super(MultiHeteroGATLayer, self).__init__()
        
        self._num_heads = num_heads
        self._merge = merge
        
        if self._merge == 'cat':
            self.gat_conv = HeteroGATLayer(in_dim, out_dim, cetypes, num_heads)
        else:
            self.gat_conv = HeteroGATLayer(in_dim, out_dim, cetypes, num_heads, use_relu = False)              

    def forward(self, g, feat_dict):
        tmp = self.gat_conv(g, feat_dict)
        results = {}
       
        if self._merge == 'cat':
            # concat on the output feature dimension (dim=1)  
            for ntype in g.ntypes:
                results[ntype] = torch.flatten(tmp[ntype], 1)
        else:
            # merge using average
            for ntype in g.ntypes:
                results[ntype] = torch.mean(tmp[ntype], 1)
        
        return results
