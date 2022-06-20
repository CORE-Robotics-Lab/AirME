# -*- coding: utf-8 -*-
"""
different heuristics for repair scheduling
"""

import random
import copy

'''
Schedule randomly
'''
class RandomScheduler(object):
    
    def __init__(self, num_planes = 20, num_crews = 2):
        self.num_planes = num_planes
        self.num_crews = num_crews
    
    # pick one action
    # pick random from all crews and planes
    # 0: don't schedule
    def get_schedule_action(self):
        crew_no = random.randint(1, self.num_crews)
        plane_no = random.randint(0, self.num_planes)
        
        return crew_no, plane_no
    
    # pick one action
    # pick random from available crews & planes
    # 0: don't schedule
    def get_available_action(self, crews, planes):
        if len(crews) <= 0 or len(planes) <= 0:
            return 0, 0
        
        #crew_idx = random.randint(0, len(crews)-1)
        #crew_no = crews[crew_idx]
        crew_no = random.choice(crews)

        planes.append(0)
        #plane_idx = random.randint(0, len(planes)-1)
        #plane_no = planes[plane_idx]
        plane_no = random.choice(planes)
        
        return crew_no, plane_no

    # pick several actions
    # for every available crew, pick a plane, or 0
    # return crew list and plane list in the same order
    # note that len(planes_avail) is always > len(crews_avail) for this case
    def get_multi_available_action(self, crews, planes):
        if len(crews) <= 0 or len(planes) <=0:
            return [], []
        
        crew_list = crews.copy()
        #plane_list = random.sample(planes, k=len(crews))
        plane_list = []
        planes.append(0)
        for i in range(len(crews)):
            plane_no = random.choice(planes)
            plane_list.append(plane_no)
            if plane_no != 0:
                planes.remove(plane_no)
        
        return crew_list, plane_list
    
    # pick several actions
    # for every broken plane, try to assign an available crew
    def get_fix_broken(self, crews, planes):
        if len(crews) <= 0 or len(planes) <=0:
            return [], []

        crew_list = []
        plane_list = []
        for i in range(len(planes)):
            #plane_no = planes[i]
            #plane_list.append(plane_no)
            if i < len(crews):
                crew_list.append(crews[i])
                plane_list.append(planes[i][0])

        return crew_list, plane_list
    
    # single action version
    def get_fix_broken_single(self, crews, planes):
        if len(crews) <= 0 or len(planes) <=0:
            return 0, 0
        
        plane_picked = random.choice(planes)
        
        return crews[0], plane_picked[0]


'''
Condition-based Scheduler
'''
class HybridScheduler(object):

    def __init__(self, num_planes = 20, num_crews = 2):
        self.num_planes = num_planes
        self.num_crews = num_crews
    
    # pick several actions
    def get_schedule_actions(self, crews, planes_broken, planes_avail,
                             th_type, th_value):
        if len(crews) <= 0:
            return [], []
        
        crew_list = crews.copy()
        plane_list = []
        
        # rank broken planes based on broken days
        #planes_broken.sort(key=lambda x: x[1], reverse=True)
        # rank broken planes based on flying reward
        if len(planes_broken) > 0:
            planes_broken.sort(key=lambda x: x[2], reverse=True)
        
        '''
        rank available planes if needed
            based on operating hour
            or a failure prediction model
            also set a threshold for entering the queue
        '''
        # threshold_hours = 40
        # threshold_landings = 10
        # threshold_prob = 0.1
        # calculate failure probability
        # failure_prob = self.failure_prediction(planes_avail)
        
        if len(crews) > len(planes_broken):
            planes_avail_list = []
            for i in range(len(planes_avail)):
                if th_type == 'hour':
                    if planes_avail[i][1] >= th_value:
                        planes_avail_list.append(planes_avail[i])
                elif th_type == 'landing':
                    if planes_avail[i][2] >= th_value:
                        planes_avail_list.append(planes_avail[i])
                elif th_type == 'prob':
                    if planes_avail[i][4] >= th_value:
                        planes_avail_list.append(planes_avail[i])
        
            if len(planes_avail_list) > 0:
                if th_type == 'hour':
                    planes_avail_list.sort(key=lambda x: x[1], reverse=True)
                elif th_type == 'landing':
                    planes_avail_list.sort(key=lambda x: x[2], reverse=True)
                elif th_type == 'prob':
                    planes_avail_list.sort(key=lambda x: x[4], reverse=True)
            
        # if crews not enough, pick in order or random from brokens
        #      this can be done by shuffling planes_broken beforehand
        #      or rank the broken planes according to some rules
        # else 1. repair all broken planes 
        #      2. and use the rest of crews for available planes
        for i in range(len(crews)):
            # assign broken planes
            if i < len(planes_broken):
                plane_list.append(planes_broken[i][0])
            # if there are still crews available after assigning all broken planes
            # randomly pick available planes or 0: don't schedule or sort by sth
            elif i < len(planes_broken) + len(planes_avail_list):
                plane_list.append(planes_avail_list[i-len(planes_broken)][0])
#                plane_no = random.choice(planes_avail_id)
#                plane_list.append(plane_no)
#                if plane_no != 0:
#                    planes_avail_id.remove(plane_no)
            else:
                plane_list.append(0)
        
        return crew_list, plane_list
    
    # single action version 
    # make use of get_schedule_actions
    def get_single_action(self, crews, planes_broken, planes_avail,
                          th_type, th_value):
        if len(crews) <= 0:
            return 0, 0
        
        crew_list, plane_list = self.get_schedule_actions(crews, planes_broken,
                                                          planes_avail,
                                                          th_type, th_value)

        return crew_list[0], plane_list[0]
    
'''
Helper function for model-based planning
Compute the joint failure probability of each avail plane
Have full access to the environment failure function/model + plane type
    r_env: repair environment
Return
    planes_avail list with prob        
'''
def get_avail_with_prob(r_env):
    planes_avail = []   
    # check planes
    for i in range(r_env.num_planes):
        if r_env.planes[i].task.crew == 0:
            if not r_env.planes[i].is_broken:
                tmp_p = copy.deepcopy(r_env.planes[i])
                prob_not_break = 1.0
                for j in range(tmp_p.num_parts):
                    if j == 0:
                        rt, prob = tmp_p.parts[j].check_break(tmp_p.num_landings, 0)
                    else:
                        rt, prob = tmp_p.parts[j].check_break(tmp_p.operating_hours/tmp_p.hour_scale,
                                                              5.0 / tmp_p.hour_scale)
                    prob_not_break = prob_not_break * (1 - prob)
                
                prob_break = 1.0 - prob_not_break   

                planes_avail.append([r_env.planes[i].id,
                                     r_env.planes[i].operating_hours,
                                     r_env.planes[i].num_landings,
                                     r_env.planes[i].flying_reward,
                                     prob_break])
    
    return planes_avail   

'''
Time-based Scheduler
'''
class PeriodicScheduler(object):
    
    def __init__(self, num_planes = 20, num_crews = 2, interval = 40):
        self.num_planes = num_planes
        self.num_crews = num_crews
        self.interval = interval

    # pick several actions
    def get_schedule_actions(self, crews, planes_broken, planes_avail, current_time):
        if len(crews) <= 0:
            return [], []
        
        # only schedule every given interval
        if current_time % self.interval != 0:
            return [], []
        
        crew_list = crews.copy()
        plane_list = []        

        # rank broken planes based on flying reward
        if len(planes_broken) > 0:
            planes_broken.sort(key=lambda x: x[2], reverse=True)

        if len(crews) > len(planes_broken):
            planes_avail_list = []
            for i in range(len(planes_avail)):
                planes_avail_list.append(planes_avail[i])
        
            if len(planes_avail_list) > 0:
                planes_avail_list.sort(key=lambda x: x[1], reverse=True)

        for i in range(len(crews)):
            # assign broken planes
            if i < len(planes_broken):
                plane_list.append(planes_broken[i][0])
            # if there are still crews available after assigning all broken planes
            # pick available planes sort by sth
            elif i < len(planes_broken) + len(planes_avail_list):
                plane_list.append(planes_avail_list[i-len(planes_broken)][0])
            else:
                plane_list.append(0)
        
        return crew_list, plane_list
