# -*- coding: utf-8 -*-
"""
AirME and helper functions

1. time step is hour-based
2. add null-plane node for denoting not selecting any plane
    null-plane is added as plane node#0
3. V2 simplifed graph layer
"""

import random
import math
import copy

import numpy as np
from scipy.special import gamma

import dgl
import torch

'''
Generate initial node features for hetgraph
    num_planes_avail: does not include null plane#0
        it denotes num of planes not under repair
    normalization/preprocessing added
'''
def hetgraph_node_helper(num_planes, planes, num_crews, crews,
                         num_planes_avail, num_crews_avail,
                         num_airliners = 25,
                         total_reward = 100.0):
    feat_dict = {}

    feat_dict['plane'] = np.zeros((num_planes+1, 14), dtype=np.float32)
    for i in range(num_planes):
        # idx = 0 is reserved for null plane#0
        # features from plane states
        feat_dict['plane'][i+1,0] = planes[i].operating_hours / 10.0
        feat_dict['plane'][i+1,1] = planes[i].prob_use
        feat_dict['plane'][i+1,2] = planes[i].num_landings / 5.0
        if planes[i].is_broken:
            feat_dict['plane'][i+1,3] = 1
        feat_dict['plane'][i+1,4] = planes[i].flying_reward
        # assoicated task features if under repair
        if planes[i].task.crew > 0:
            feat_dict['plane'][i+1,5] = 1
            feat_dict['plane'][i+1,6] = planes[i].task.duration / 10.0
            feat_dict['plane'][i+1,7] = planes[i].task.cost / planes[i].task.duration
            feat_dict['plane'][i+1,8] = planes[i].task.progress / 10.0
        # mission features
        if planes[i].is_flying:
            feat_dict['plane'][i+1,9] = 1
            feat_dict['plane'][i+1,10] = planes[i].mission.duration / 10.0
            feat_dict['plane'][i+1,11] = planes[i].mission.progress / 10.0
        # [1 0] for airliners, [0 1] for helicopters
        if i < num_airliners:
            feat_dict['plane'][i+1,12] = 1
        else:
            feat_dict['plane'][i+1,13] = 1
        
    # feature for null plane#0
    # pesudo plane features
    feat_dict['plane'][0, 0] = 1 / 10.0 #operating_hours
    # prob_use & flying_reward & num_landings are 0
    # broken
    feat_dict['plane'][0, 3] = 1
    # not under repair & not flying
    # one-hot encoding [0 0] for plane#0
    
    feat_dict['crew'] = np.zeros((num_crews, 4), dtype=np.float32)
    for i in range(num_crews):
        if crews[i].plane == 0:
            feat_dict['crew'][i,0] = 1
        else:
            feat_dict['crew'][i,1] = 1
            plane_no = crews[i].plane
            task_dur = planes[plane_no-1].task.duration
            feat_dict['crew'][i,2] = task_dur / 10.0
            feat_dict['crew'][i,3] = crews[i].progress / 10.0
        
    feat_dict['state'] = np.array((num_planes, num_crews, 
                                   num_planes_avail, num_crews_avail,
                                   total_reward), 
                                  dtype=np.float32).reshape(1, 5)
        
    # Q score nodes
    feat_dict['value'] = np.zeros((num_planes_avail+1, 1), dtype=np.float32)
    
    return feat_dict

'''
Helper function for building HetGraph
Score nodes are built w.r.t planes not under repair
    In this version, no edges between plane nodes
        no edges between crew nodes
'''
def build_hetgraph(num_planes, num_crews, crew_plane_list, 
                   planes_not_under_repair,
                   device = torch.device('cuda')):
    
    data_dict = {}
    
    # 1. No edges between plane nodes; no edges between crew nodes
        
    # 2+. Edges between crew and plane
    # crew id & plane id start from #1
    # crew_plane_list = [[1,2],[2,4],[3,4]]
    # [crew] — [repairing] — [plane]
    crp_u = []
    crp_v = []
    # [plane] — [repaired_by] — [crew]
    # null plane#0 not connected with any crew
    for i in range(len(crew_plane_list)):
        crp_u.append(crew_plane_list[i][0]-1)
        crp_v.append(crew_plane_list[i][1])
    
    data_dict[('crew','repairing','plane')] = (crp_u, crp_v)
    data_dict[('plane','repaired_by','crew')] = (crp_v, crp_u)

    # 3. graph summary node / state
    #[plane] — [p_in] — [state]
    data_dict[('plane','p_in','state')] = (list(range(num_planes+1)),
                                           [0 for i in range(num_planes+1)])
    
    #[crew] — [c_in] — [state]   
    data_dict[('crew','c_in','state')] = (list(range(num_crews)),
                                          [0 for i in range(num_crews)])
    
    #[state] — [s_in] — [state]
    data_dict[('state','s_in','state')] = ([0], [0])

    # 4. Q score/value node
    # q(state, p) | p are planes not under repair, always including plane#0
    planes = np.array(planes_not_under_repair, dtype=np.int64)     
    # add plane#0 at the end of plane list
    planes = np.append(planes, 0)
    num_scores = len(planes)
    #[plane] — [p_to]—[value]
    data_dict[('plane','p_to','value')] = (planes, list(range(num_scores)))
    
    #[state] — [s_to]—[value]    
    data_dict[('state','s_to','value')] = ([0 for i in range(num_scores)],
                                           list(range(num_scores)))
    
    #[value] — [v_to] — [value]    
    data_dict[('value','v_to','value')] = (list(range(num_scores)),
                                           list(range(num_scores)))   
    
    num_nodes_dict = {'plane': num_planes + 1,
                      'crew': num_crews,
                      'state': 1,
                      'value': num_scores}

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)#, idtype=torch.int64)
    g = g.to(device)

    return g


def MD_build_hetgraph(num_planes, num_crews, crew_plane_list, 
                      planes_not_under_repair, crews_scheduled, planes_scheduled,
                      device = torch.device('cuda')):
    """
    Build HetGraph for pick_plane in Multi Decision Scheduler
            
        crews_scheduled, planes_scheduled
            store decisions made in current time step
    """    
    data_dict = {}
    
    # 1. No edges between plane nodes; no edges between crew nodes
        
    # 2+. Edges between crew and plane
    # crew id & plane id start from #1
    # crew_plane_list = [[1,2],[2,4],[3,4]]
    # [crew] — [repairing] — [plane]
    crp_u = []
    crp_v = []
    # [plane] — [repaired_by] — [crew]
    # null plane#0 not connected with any crew
    for i in range(len(crew_plane_list)):
        crp_u.append(crew_plane_list[i][0]-1)
        crp_v.append(crew_plane_list[i][1])
    
    # also add the decisions made so far in current time step
    for j in range(len(crews_scheduled)):
        crp_u.append(crews_scheduled[j]-1)
        crp_v.append(planes_scheduled[j])
    
    data_dict[('crew','repairing','plane')] = (crp_u, crp_v)
    data_dict[('plane','repaired_by','crew')] = (crp_v, crp_u)

    # 3. graph summary node / state
    #[plane] — [p_in] — [state]
    data_dict[('plane','p_in','state')] = (list(range(num_planes+1)),
                                           [0 for i in range(num_planes+1)])
    
    #[crew] — [c_in] — [state]   
    data_dict[('crew','c_in','state')] = (list(range(num_crews)),
                                          [0 for i in range(num_crews)])
    
    #[state] — [s_in] — [state]
    data_dict[('state','s_in','state')] = ([0], [0])

    # 4. Q score/value node
    # q(state, p) | p are planes not under repair, always including plane#0
    planes = np.array(planes_not_under_repair, dtype=np.int64)     
    # add plane#0 at the end of plane list
    planes = np.append(planes, 0)
    num_scores = len(planes)
    #[plane] — [p_to]—[value]
    data_dict[('plane','p_to','value')] = (planes, list(range(num_scores)))
    
    #[state] — [s_to]—[value]    
    data_dict[('state','s_to','value')] = ([0 for i in range(num_scores)],
                                           list(range(num_scores)))
    
    #[value] — [v_to] — [value]    
    data_dict[('value','v_to','value')] = (list(range(num_scores)),
                                           list(range(num_scores)))   
    
    num_nodes_dict = {'plane': num_planes + 1,
                      'crew': num_crews,
                      'state': 1,
                      'value': num_scores}

    g = dgl.heterograph(data_dict, num_nodes_dict=num_nodes_dict)#, idtype=torch.int64)
    g = g.to(device)

    return g

def MD_node_helper(num_planes, planes, num_crews, crews,
                   num_planes_avail, num_crews_avail,
                   crews_scheduled, planes_scheduled,
                   num_airliners, total_reward):
    """
    Generate initial node features for pick_plane in Multi Decision Scheduler
        num_planes_avail: does not include null plane#0
            it denotes num of planes not under repair
        normalization/preprocessing added
        
        crews_scheduled, planes_scheduled
            store decisions made in current time step
    """
    feat_dict = {}

    feat_dict['plane'] = np.zeros((num_planes+1, 16), dtype=np.float32)
    
    for i in range(num_planes):
        # idx = 0 is reserved for null plane#0
        
        # features from plane states
        feat_dict['plane'][i+1,0] = planes[i].operating_hours / 10.0
        feat_dict['plane'][i+1,1] = planes[i].prob_use
        feat_dict['plane'][i+1,2] = planes[i].num_landings / 5.0
        
        if planes[i].is_broken:
            feat_dict['plane'][i+1,3] = 1
            
        feat_dict['plane'][i+1,4] = planes[i].flying_reward
        
        # assoicated task features if under repair
        if planes[i].task.crew > 0:
            # under repair / schedule in previous time steps
            feat_dict['plane'][i+1,6] = 1
            feat_dict['plane'][i+1,8] = planes[i].task.duration / 10.0
            feat_dict['plane'][i+1,9] = planes[i].task.cost / planes[i].task.duration
            feat_dict['plane'][i+1,10] = planes[i].task.progress / 10.0
        elif planes[i].id in planes_scheduled:
            # Scheduled in current time step (so task cost and duration are N/A)
            feat_dict['plane'][i+1,7] = 1
        else:
            # Unscheduled / not under repair
            feat_dict['plane'][i+1,5] = 1       
            
        # mission features
        if planes[i].is_flying:
            # check if the plane has been scheduled in current time step
            if planes[i].id not in planes_scheduled:         
                feat_dict['plane'][i+1,11] = 1
                feat_dict['plane'][i+1,12] = planes[i].mission.duration / 10.0
                feat_dict['plane'][i+1,13] = planes[i].mission.progress / 10.0
        
        # [1 0] for airliners, [0 1] for helicopters
        if i < num_airliners:
            feat_dict['plane'][i+1,14] = 1
        else:
            feat_dict['plane'][i+1,15] = 1
        
    # feature for null plane#0
    # pesudo plane features
    feat_dict['plane'][0, 0] = 1 / 10.0 #operating_hours
    # prob_use & flying_reward & num_landings are 0
    # broken
    feat_dict['plane'][0, 3] = 1
    # not under repair & not flying
    feat_dict['plane'][0, 5] = 1
    # one-hot encoding [0 0] for plane#0
    
    feat_dict['crew'] = np.zeros((num_crews, 5), dtype=np.float32)
    for i in range(num_crews):
        if crews[i].id in crews_scheduled:
            feat_dict['crew'][i,2] = 1
        elif crews[i].plane == 0:
            feat_dict['crew'][i,0] = 1
        else:
            feat_dict['crew'][i,1] = 1
            plane_no = crews[i].plane
            task_dur = planes[plane_no-1].task.duration
            feat_dict['crew'][i,3] = task_dur / 10.0
            feat_dict['crew'][i,4] = crews[i].progress / 10.0
        
    feat_dict['state'] = np.array((num_planes, num_crews, 
                                   num_planes_avail, num_crews_avail,
                                   total_reward), 
                                  dtype=np.float32).reshape(1, 5)
        
    # Q score nodes
    feat_dict['value'] = np.zeros((num_planes_avail+1, 1), dtype=np.float32)
    
    return feat_dict


# define the pdf of Weibull distribution
# k - shape
# lambda - scale
def weib(x, scale, shape):
    return (shape / scale) * (x / scale)**(shape - 1) * np.exp(-(x / scale) ** shape)

'''
Part class
'''
class Part(object):
    
    def __init__(self, p_id, scale, shape):
        self.id = p_id
        self.scale = scale
        self.shape = shape
    
    def cal_prob(self, usage):
        prob = weib(usage, self.scale, self.shape)
        return prob
    
    def check_break(self, usage, threshold):
        # part won't break for the first thresholds hours/usages
        if usage <= threshold:
            return False, 0.0
        
        prob = self.cal_prob(usage)
        if random.random() <= prob:
            return True, prob
        else:
            return False, prob

    # mean of the Weibull distribution
    def cal_mean(self):
        wei_mean = self.scale * gamma(1 + 1/self.shape)
        return wei_mean

'''
Task class
'''
class RepairTask(object):
    
    def __init__(self, t_id, repair_duration = 0, repair_cost = 0.0, 
                 progress = 0, crew = 0):
        self.id = t_id
        self.duration = repair_duration
        self.cost = repair_cost
        # specify the task progress in time
        self.progress = progress
        # 0: no crew, 1-M: crew number
        self.crew = crew
        
'''
Operation mission task
'''
class FlyMission(object):
    
    def __init__(self, f_id, mission_duration = 0, progress = 0):
        self.id = f_id
        # dur = 0 means no mission
        self.duration = mission_duration
        self.progress = 0

'''
Airplane class
    1-N
'''
class Airplane(object):
    
    def __init__(self, a_id, num_parts, scale_list, shape_list, prob_use, hour_scale,
                 operating_hours = 0, is_broken = False,
                 is_flying = False, flying_reward = None,
                 num_landings = 0):
        '''
        parameters
        '''
        self.id = a_id
        # running time since last repair
        self.operating_hours = operating_hours
        # True -- Operating/flying; False -- Grounded/park
        self.is_flying = is_flying
        self.mission = FlyMission(self.id)
        # broken or not
        self.is_broken = is_broken
        self.hour_since_broken = 0
        # probability of being used at beginning of the next hour when grounded
        self.prob_use = prob_use
        self.hour_scale = hour_scale
        
        # number of landings
        self.num_landings = num_landings
        
        '''
        associated task
        '''
        self.task = RepairTask(self.id)
        # randomized operating reward per hour
        if flying_reward is None:
            self.flying_reward = random.uniform(1, 20) 
        else:
            self.flying_reward = flying_reward  
        '''
        buffer for break pattern sequence
        '''
        self.seq = []
        self.seq_landings = []
        self.counter = 0    #count the number of breaks happened so far

        '''
        components/parts p0-p{n-1}
        '''
        self.num_parts = num_parts
        self.parts = []
        for i in range(self.num_parts):
            self.parts.append(Part(i, scale_list[i], shape_list[i])) 

    # The prediction model used to decide
    # when the plane will break based on its parameters.
    # run this at the beginning of each time step
    def check_break(self, fix_break = False):
        # only check planes in mission
        if self.is_flying:
            # use saved break sequence
            if fix_break:
                if self.operating_hours >= self.seq[self.counter]:
                    self.is_broken = True
                    return True
                elif self.num_landings >= self.seq_landings[self.counter]:
                    self.is_broken = True
                    return True
                else:
                    self.is_broken = False
                    return False
            # use real-time 
            else:                          
                for i in range(self.num_parts):
                    if i == 0:
                        '''
                        new: the first part uses num_landings
                        '''
                        failure, prob = self.parts[i].check_break(self.num_landings, 0)
                    else:
                        failure, prob = self.parts[i].check_break(self.operating_hours/self.hour_scale,
                                                                  5.0 / self.hour_scale)
                    
                    if failure:
                        self.is_broken = True
                        # stop mission
                        self.is_flying = False
                        self.mission.duration = 0
                        self.mission.progress = 0
                        return True
            
                return False
        else:
            # return unchanged
            return self.is_broken            

    # The duration a repair task takes is drawn from an estimated distribution,
    # and is also affected by the state of the plane when repair starts.
    # if breaks, add another amount of time
    def generate_repair_duration(self, add_break_time = 12):
        self.task.duration = random.randint(2, 8)
        
        # factor in the plane status
        self.task.duration += int(self.operating_hours/24)
        self.task.duration += int(self.num_landings/6)

        if self.is_broken:
            self.task.duration = self.task.duration + add_break_time

    # Calculates repair cost
    # if breaks, add break cost
    def generate_repair_cost(self, add_break_cost = 48.0):
        base_cost = random.uniform(0.1, 1) * self.flying_reward
        labor_cost = 2 * self.task.duration
        add_cost = 0.0
        # break cost
        if self.is_broken:
            self.task.cost = base_cost + labor_cost + add_break_cost + add_cost
        else:
            self.task.cost = base_cost + labor_cost + add_cost
    
    # update plane parameters and task status by moving one hour forward
    def update(self):
        if self.task.crew == 0:
            # update when not under repair and not broken
            if not self.is_broken:
                # flying
                if self.is_flying:
                    self.operating_hours += 1
                    self.mission.progress += 1
            # update when not under repair but broken
            else:
                self.hour_since_broken += 1
        else:
            # update when under repair
            self.task.progress += 1

    # release (the completed) task or (the completed) fly mission
    # and reset plane
    def reset(self):
        # reset - completing repair task
        # broken -> grounded
        if self.task.crew > 0:
            # release task
            self.task.crew = 0
            self.task.progress = 0
            # reset plane
            self.operating_hours = 0
            self.is_flying = False
            self.is_broken = False
            self.hour_since_broken = 0
            self.num_landings = 0
            # update break counter
            # increase counter by 1 after each repair
            self.counter = self.counter + 1
            if self.counter >= len(self.seq):
                self.counter = 0
        # reset - completing fly mission
        # flying -> grounded
        elif self.mission.duration > 0:
            # release mission
            self.mission.duration = 0
            self.mission.progress = 0
            # ground plane
            self.is_flying = False
            self.num_landings += 1

'''
Repair Crew class
    1-M
'''
class Crew(object):
    
    def __init__(self, c_id, plane=0, progress=0):
        self.id = c_id
        # 0: not working, 1-N: plane id
        self.plane = plane
        self.progress = progress
    
    def update(self):
        if self.plane > 0:
            self.progress += 1
            
    def reset(self):
        self.plane = 0
        self.progress = 0

'''
Environment
'''
class RepairEnv(object):
    
    '''
    Parameters on plane components/parts
        num_parts_list: list of num_parts for each plane
        scale_lists: list of scale_list for each plane
        shape_lists: list of shape_list for each plane
        prob_uses: list of prob_use for each plane
        hour_scales: list of hour_scale for each plane
        hours_list: list of initial operating hours for each plane
        num_landings_list: list of num_landings for each plane
        broken_list: list of True/False if a plane is broken or not
        reward_list: flying reward for each plane
    '''
    def __init__(self, num_airliners, num_helicopters, num_parts_list, scale_lists, 
                 shape_lists, prob_uses, hour_scales,
                 hours_list, num_landings_list, broken_list, reward_list = None,
                 num_crews = 2):
        self.num_airliners = num_airliners
        self.num_helicopters = num_helicopters
        self.num_planes = num_airliners + num_helicopters
        
        self.planes = []
        for i in range(self.num_planes):
            if reward_list is None:
                self.planes.append(Airplane(i+1, num_parts_list[i], scale_lists[i],
                                            shape_lists[i], prob_uses[i],
                                            hour_scales[i],
                                            operating_hours = hours_list[i],
                                            is_broken = broken_list[i],
                                            num_landings = num_landings_list[i]))
            else:
                self.planes.append(Airplane(i+1, num_parts_list[i], scale_lists[i],
                                            shape_lists[i], prob_uses[i],
                                            hour_scales[i], 
                                            operating_hours = hours_list[i],
                                            is_broken = broken_list[i],
                                            num_landings = num_landings_list[i],
                                            flying_reward = reward_list[i]))                

        self.num_crews = num_crews
        self.crews = [Crew(i) for i in range(1, num_crews+1)]
                               
        # hour-based time system
        self.total_hours = 0
        self.total_days = 0
        self.current_hour = 0
        
        # total and average flying_reward (per plane)
        # used to normlize reward & cost
        rewards = []
        for i in range(self.num_planes):
            rewards.append(self.planes[i].flying_reward)
            
        self.total_fr = np.sum(np.array(rewards))
        self.avg_fr = np.mean(np.array(rewards))
    
    # update environment with given scheduling action
    # 0 means NULL
    def step(self, crew_no = 0, plane_no = 0, verbose = True, norm_r = False,
             fix_break = False):
        # perform scheduling action
        if crew_no > 0 and plane_no > 0:
            self.send_crew(crew_no, plane_no)
        
        # sample usage
        self.sample_usage()
        
        # check whether plane breaks
        self.check_break(fix_break = fix_break)
        
        # update plane and task status
        self.update_planes_and_tasks() 
        
        # update crew status
        self.update_crews()        
        
        # collect cost & reward, plus # of planes avail (not broken or under repair)
        cost, reward, avail = self.collect_cost_reward(verbose) 
        
        # update time system
        self.total_hours += 1
        self.current_hour += 1
        if self.current_hour >= 24:
            self.total_days += 1
            self.current_hour = 0
        
        # release completed tasks and missions
        self.release_completed()
        
        # normalize cost & reward when needed
        if norm_r:
            cost = cost / self.total_fr
            reward = reward / self.total_fr
        
        return cost, reward, avail

    # take several scheduling decisions and update all related crews & planes
    def step_multi(self, crew_list, plane_list, verbose = True, norm_r = False,
                   fix_break = False):
        # perform scheduling action
        for i in range(len(crew_list)):
            crew_no = crew_list[i]
            plane_no = plane_list[i]
            
            if crew_no > 0 and plane_no > 0:
                self.send_crew(crew_no, plane_no)        
        
        # sample usage
        self.sample_usage()
        
        # check whether plane breaks
        self.check_break(fix_break = fix_break)
        
        # update plane and task status
        self.update_planes_and_tasks() 
        
        # update crew status
        self.update_crews()        
        
        # collect cost & reward, plus # of planes avail (not broken or under repair)
        cost, reward, avail = self.collect_cost_reward(verbose) 
        
        # update time system
        self.total_hours += 1
        self.current_hour += 1
        if self.current_hour >= 24:
            self.total_days += 1
            self.current_hour = 0
        
        # release completed tasks and missions
        self.release_completed()
        
        # normalize cost & reward when needed
        if norm_r:
            cost = cost / self.total_fr
            reward = reward / self.total_fr
        
        return cost, reward, avail        
        
            
    def set_time(self, set_day, set_hour):
        self.total_days = set_day
        self.current_hour = set_hour
        self.total_hours = set_day * 24 + set_hour
        
    # At the beginning of each hour, for every grounded plane
    # that is not broken and not under repair
    # based on its prob_use, sample to see if it will be used for operation
    # if yes, generate that operation info
    def sample_usage(self):
        for i in range(self.num_planes):
            if not self.planes[i].is_flying:
                if not self.planes[i].is_broken:
                    if self.planes[i].task.crew == 0:
                        # use its prob_use to sample
                        if random.random() <= self.planes[i].prob_use:
                            # start a flying operation
                            mision_dur = random.randint(5, 10)
                            self.planes[i].mission.duration = mision_dur
                            self.planes[i].is_flying = True

    # check whether plane breaks
    def check_break(self, fix_break = False):
        for i in range(self.num_planes):
            #print(self.planes[i].id)
            self.planes[i].check_break(fix_break = fix_break)
    
    # perfrom a scheduling decision
    def send_crew(self, crew_no, plane_no):       
        if crew_no > self.num_crews or plane_no > self.num_planes:
            print('Invalid number')
            return False    

        if self.crews[crew_no-1].plane > 0:
            print('Crew not available')
            return False
        
        if self.planes[plane_no-1].task.crew > 0:
            print('Plane not available')
            return False
        
        # assign plane to crew
        self.crews[crew_no-1].plane = plane_no
        self.crews[crew_no-1].progress = 0
        
        # get repair duration/cost and relate crew to plane/task
        self.planes[plane_no-1].generate_repair_duration()
        self.planes[plane_no-1].generate_repair_cost()        
        self.planes[plane_no-1].task.crew = crew_no
        self.planes[plane_no-1].task.progress = 0
        
        # if plane is in mission, stop the mission and get grounded
        self.planes[plane_no-1].is_flying = False
        self.planes[plane_no-1].mission.duration = 0
        self.planes[plane_no-1].mission.progress = 0

    # update task status
    def update_planes_and_tasks(self):
        for i in range(self.num_planes):
            self.planes[i].update()
    
    # update crew status       
    def update_crews(self):
        for i in range(self.num_crews):
            self.crews[i].update()

    # collect cost and reward
    # negative reward for repair cost
    # positive reward for letting airplane fly without repairing
    # also print the status of each plane
    def collect_cost_reward(self, verbose = True):
        cost = 0.0
        reward = 0.0
        break_count = 0
        avail_count = 0
        
        for i in range(self.num_planes):
            if self.planes[i].task.crew > 0:
                if verbose:
                    print('Plane %02d is under repair' % self.planes[i].id)
                cost = cost + self.planes[i].task.cost / self.planes[i].task.duration
            elif self.planes[i].is_broken:
                if verbose:
                    print('Plane %02d breaks and needs repair' % self.planes[i].id)
                break_count += 1
            elif self.planes[i].is_flying:
                if verbose:
                    print('Plane %02d is operating' % self.planes[i].id)
                reward = reward + self.planes[i].flying_reward
                avail_count += 1
                #print('reward added %f' % reward)
            else:
                if verbose:
                    print('Plane %02d is grounded' % self.planes[i].id)
                avail_count += 1
                
        return cost, reward, avail_count

    # release completed tasks and missions
    def release_completed(self):
        # check completed tasks
        for i in range(self.num_crews):
            p_no = self.crews[i].plane
            if p_no > 0:
                if self.crews[i].progress == self.planes[p_no-1].task.duration:
                    #print('releasing task %d' % p_no)
                    # release crew
                    self.crews[i].reset()
                    # release task and reset plane
                    self.planes[p_no-1].reset()
                    #print('Plane %02d has been reset' % self.planes[i].id)

        # check completed missions
        for i in range(self.num_planes):
            if self.planes[i].is_flying:
                if self.planes[i].mission.progress == self.planes[i].mission.duration:
                    #print('Plane %02d mission finished' % self.planes[i].id)
                    # reset plane
                    self.planes[i].reset()

    # return available crews and available planes (=not under repair)
    # return id list
    def get_available(self):
        crews_avail = self.get_available_crews()
        planes_avail = self.get_planes_not_under_repair()
        
        return crews_avail, planes_avail
    
    # return available crews
    # return id list
    def get_available_crews(self):
        crews_avail = []
        # check crews
        for i in range(self.num_crews):
            if self.crews[i].plane == 0:
                crews_avail.append(self.crews[i].id)
        
        return crews_avail   
    
    # return broken planes not under repair
    # also return available planes not broken + not under repair
    # return info list
    def get_broken_and_available(self):
        planes_broken = []
        planes_avail = []
        # check planes
        for i in range(self.num_planes):
            if self.planes[i].task.crew == 0:
                if self.planes[i].is_broken:
                    planes_broken.append([self.planes[i].id,
                                          self.planes[i].hour_since_broken,
                                          self.planes[i].flying_reward])
                else:
                    planes_avail.append([self.planes[i].id,
                                         self.planes[i].operating_hours,
                                         self.planes[i].num_landings,
                                         self.planes[i].flying_reward])
                    
        return planes_broken, planes_avail

    # return working crews and their assigned planes in a list
    def get_crew_plane_list(self):
        cplist = []
        
        for i in range(self.num_crews):
            if self.crews[i].plane > 0:
                cplist.append([self.crews[i].id, self.crews[i].plane])
        
        return cplist
    
    # return planes not under repair
    def get_planes_not_under_repair(self):
        plist = []
        
        for i in range(self.num_planes):
            if self.planes[i].task.crew == 0:
                plist.append(self.planes[i].id)
                
        return plist
    
    # get the percentage of broken planes (including those under repair)
    def get_broken_rate(self):
        num_broken = 0.0
        for i in range(self.num_planes):
            if self.planes[i].is_broken:
                num_broken += 1   
        
        return num_broken / self.num_planes
    
    # called right after the env is initialized
    def sample_break_sequence(self, max_no):
        self.max_no = max_no
        for i in range(self.num_planes):
            # make a deep copy for break sampling
            tmp_p = copy.deepcopy(self.planes[i])
            # the plane is always flying
            tmp_p.is_flying = True
            # starts with 0
            tmp_p.operating_hours = 0
            tmp_p.num_landings = 0
            
            for j in range(self.max_no):                
                # simulate forward untill plane breaks                    
                while not tmp_p.is_broken:
                    tmp_p.operating_hours += 1
                    # use mean mission duraiton (7.5) for num_landings
                    tmp_p.num_landings = round(tmp_p.operating_hours/7.5)
                    tmp_p.check_break()
                # record break hour                
                self.planes[i].seq.append(tmp_p.operating_hours)
                self.planes[i].seq_landings.append(tmp_p.num_landings)
                # reset
                tmp_p.is_broken = False
                tmp_p.is_flying = True
                tmp_p.operating_hours = 0
                tmp_p.num_landings = 0


def get_default_param(num_airliners, num_helicopters, random_init = False,
                      percent_broken = -1.0):
    plane_info = {}
    
    '''
    Initialization of plane parameters
    '''
    plane_info['num_parts_list'] = []
    plane_info['scale_lists'] = []
    plane_info['shape_lists'] = []
    plane_info['prob_uses'] = []
    plane_info['hour_scales'] = []
    plane_info['reward_list'] = []
    
    # first airliners
    for i in range(num_airliners):
        plane_info['num_parts_list'].append(4)
        plane_info['scale_lists'].append([15, 12, 18, 16])
        plane_info['shape_lists'].append([5 + 0.5 * i for i in range(4)])
        plane_info['prob_uses'].append(0.6)
        plane_info['hour_scales'].append(20)
        plane_info['reward_list'].append(random.uniform(1, 20))
        
    # then helicopters
    for i in range(num_helicopters):
        plane_info['num_parts_list'].append(3)
        plane_info['scale_lists'].append([8, 7, 5])
        plane_info['shape_lists'].append([7, 6, 11])
        plane_info['prob_uses'].append(0.3)
        plane_info['hour_scales'].append(15)
        plane_info['reward_list'].append(random.uniform(1, 10))
    
    '''
    Initialization of plane status
    '''
    plane_info['hours_list'] = []
    plane_info['broken_list'] = []
    plane_info['num_landings_list'] = []
          
    for i in range(num_airliners+num_helicopters):
        # randomly initialize airplane state
        if random_init:
            tmp = random.randint(0, 120)
            plane_info['hours_list'].append(tmp)
            
            tmp_landings = random.randint(math.floor(tmp/10.0), math.floor(tmp/5.0)) + random.randint(-2, 2)
            plane_info['num_landings_list'].append(max(0, tmp_landings))
        else:
            plane_info['hours_list'].append(0)
            plane_info['num_landings_list'].append(0)
        
        if percent_broken > 0.0:
            # randonly breaks
            if random.random() < percent_broken:
                plane_info['broken_list'].append(True)
            else:
                plane_info['broken_list'].append(False)
        else:
            plane_info['broken_list'].append(False)
    
    return plane_info


if __name__ == '__main__':
    plane_info = get_default_param(25, 5, random_init = True, percent_broken = 0.05)
    
    r = RepairEnv(25, 5, 
                  plane_info['num_parts_list'], plane_info['scale_lists'], 
                  plane_info['shape_lists'], plane_info['prob_uses'], plane_info['hour_scales'],
                  plane_info['hours_list'], plane_info['num_landings_list'],
                  plane_info['broken_list'], plane_info['reward_list'],
                  num_crews = 4)
    
    print(r.total_hours, r.crews[2-1].progress, r.planes[10-1].task.progress)
    r.step()
    print(r.total_hours, r.crews[2-1].progress, r.planes[10-1].task.progress)
    r.step(2,10)
    print('task duration %d' % r.planes[10-1].task.duration)
    print(r.total_hours, r.crews[2-1].progress, r.planes[10-1].task.progress)
    for i in range(10):
        # start of time step t
        r.step()
        # end of time step t
        print(r.total_hours, r.crews[2-1].progress, r.planes[10-1].task.progress)
    
    num_actions = len(r.get_planes_not_under_repair())
    # ft = hetgraph_node_helper(r.num_planes, r.planes,
    #                           r.num_crews,r.crews, num_actions)
    
    # print(ft)
    #print(r.planes[10].id, r.planes[10].operating_time, r.planes[10].tiredness)
