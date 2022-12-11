import torch
import numpy as np
import json
import pickle

WORLDS = [
            'american_football',
            'doctor_who',
            'fallout',
            'final_fantasy',
            'military',
            'pro_wrestling',
            'starwars',
            'world_of_warcraft',
            'coronation_street',
            'muppets',
            'ice_hockey',
            'elder_scrolls',
            'forgotten_realms',
            'lego',
            'star_trek',
            'yugioh'
        ]
all_types = {}
for world in WORLDS:
    print(world)
    count = 0
    categories = []
    types = pickle.load(open("../data/zeshel/documents/" + world +"_type", 'rb'))
    for key, value in types.items():
        if key in all_types:
            print(key)
        values = []
        for va in value:
            if isinstance(va, list):
                for v in va:
                    if v not in values:
                        values.append(v)
            else:
                if va not in values:
                    values.append(va)
        count += len(values)
        for va in values:
            if va not in categories:
                categories.append(va)
        all_types[key] = values
    print(len(categories))
    print(count / len(types))
pickle.dump(all_types, open("../data/zeshel/documents/all_type", 'wb'), -1)