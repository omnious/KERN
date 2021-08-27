import os
import glob
import json
import itertools
from collections import defaultdict
import pandas as pd

class InfluGroup():

    # Class for creating group combinations

    @staticmethod
    def create_permu(data, setting, use_all, replace_dict, drop_attr = []):
        
        # Create permutations of group attributes
        group_attr = defaultdict(list)
        
        if setting['location']:
            group_attr['location'] = InfluGroup.group_loc(data, replace_dict['location'])
        
        if setting['segment']:
            group_attr['segment'] = InfluGroup.group_seg(data, replace_dict['segment'])
        
        if setting['target_age']:
            group_attr['target_age'] = InfluGroup.group_age(data, replace_dict['target_age'])
        
        # If there are missing value, skip it
        if False in group_attr.values():
            return False
        
        permutations = list(itertools.product(*list(group_attr.values())))
        
        if use_all:
            permutations.append(['location:All', 'segment:All', 'target_age:All'])
        
        if drop_attr:
            permutations.extend(InfluGroup.drop_attr(group_attr, drop_attr))

        # Naming the group
        influGroup = ['__'.join(p) for p in permutations]

        return influGroup

    @staticmethod
    def group_loc(data, replace_dict = {}):
        
        if data['location']:
            locations = ['location:'+x['name'] for x in data['location']]

            # remove and replace group attributes
            if replace_dict:
                locations = InfluGroup.replace_group(locations, replace_dict)
            
            return locations
        else:
            return False
    

    @staticmethod
    def group_seg(data, replace_dict = {}):
        
        if data['segment']:
            segments = ['segment:'+data['segment'][0]['name'].split()[0].strip()]
            
            # remove and replace group attributes
            if replace_dict:
                segments = InfluGroup.replace_group(segments, replace_dict)

            return segments
        else:
            return False


    @staticmethod
    def group_age(data, replace_dict = {}):
        
        if data['target_age']:
            target_age = ['target_age:'+x['name'].split("''")[0].strip() for x in data['target_age']]

            # remove and replace group attributes
            if replace_dict:
                target_age = InfluGroup.replace_group(target_age, replace_dict)
        
            return target_age
        else:
            return False
    

    @staticmethod
    def replace_group(group, replace_dict):
        
        '''
        replace removed group to replaced group
        
        arg :
            group [list]
                - current group attributes
            replace_dict [dict]
                - key : group attribute to be removed
                - value : group attribute to be replaced

        '''        
        for remove in replace_dict.keys():
            # remove group attribute
            group = [g for g in group if not remove in g]
            
            # relace group attribute
            if not replace_dict[remove] in group:
                group.append(replace_dict[remove])
        
        return group
    

    @staticmethod
    def drop_attr(permutations, remove):
        '''
        Drop group attribute in list 'remove' and create new permutation
        
        arg
            - permutation [dict] : current group attr permutation
            - remove [list] : list of attribute to be removed
        '''
    
        # remove group must be in group attributes
        assert set(remove).issubset(set(['location', 'segment', 'target_age'])) 

        # new permutation without removed group attributes
        new_permu = []

        # pop out pop_group attribute
        for attr in permutations.keys():
            if not attr in remove:
                new_permu.append(permutations[attr])
    
        new_permu = itertools.product(*new_permu)
        return new_permu