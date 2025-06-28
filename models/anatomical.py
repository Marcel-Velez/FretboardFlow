from typing import List, Tuple, Dict
import torch
from torch.nn import Softmax, Threshold
from math import exp, sqrt
import itertools
from collections import defaultdict


SOFTMAX = Softmax(-1)
THRESHOLD_24 = Threshold(-24,0)
THRESHOLD_25 = Threshold(-25,0)
MAX_RANGE = torch.Tensor([[0  , 80 , 95, 110],
                          [80 , 0  , 52, 69 ],
                          [95 , 52 , 0 , 47 ],
                          [110, 69 , 47, 0  ]])
MIN_RANGE = torch.Tensor([[0 , 5  , 15, 25 ],
                          [5 , 0  , 6 , 12 ],
                          [15, 6  , 0 , 8.5],
                          [25, 12 ,8.5, 0  ]])
NUM_STRINGS = 6



def fret_distance(m: int, n: int, scale_length: float = 620) -> float:
    if m > n:
        m, n = n, m
    s = scale_length
    out = s*(2**(n/12) - 2**(m/12))/(2**((m+n)/12))
    return out

def string_distance(s1: int, s2: int, y: float,bridge_size: float = 58.7, nut_size: float = 44.4,
        scale_length: float = 620, num_strings: int = 6) -> float:
    string_hop = abs(s1 - s2)/(num_strings-1)
    dist = (nut_size-bridge_size)*y/scale_length + bridge_size
    return dist*string_hop

def finger_distance(s1: int, f1: int, s2: int, f2: int,
                    gamma: float = 36, max_fret: int = 24,
                    bridge_size: float = 58.7, nut_size: float = 44.4,
                    scale_length: float = 620, num_strings: int = 6) -> float:
    f_dist = fret_distance(f1, f2, scale_length)
    s_dist = string_distance(s1, s2, min(f1,f2), bridge_size, nut_size, scale_length, num_strings)
    return sqrt(f_dist**2 + s_dist**2)

def finger_score(x: float, finger1: int, finger2: int, pinky_penalty: float = 0.85) -> float:
    a = MIN_RANGE[finger1, finger2].item()
    b = MAX_RANGE[finger1, finger2].item()
    if finger1 == 3 or finger2 == 3:
        penalty = pinky_penalty
    else:
        penalty = 1
    if x < a:
        return (1 + (x - 0.99*a)**3) * penalty
    else:
        return (1 - ((x - 0.99*a)/(1.01*b - 0.99*a))**2) * penalty

def get_string_fret_set(fingering: str) -> List[Tuple[int, int]]:
    out = []
    string_count = 6
    for val in fingering.split('.'):
        if val in ['', 'x']:
            string_count -= 1
            continue
        else:
            out.append((string_count, int(val)))
            string_count -= 1
    return out

def get_possible_fingerings(sf_set) -> List[Dict[int, List[Tuple[int, int]]]]:
    out = []
    sorted_set = sorted(sorted(sf_set, key=(lambda x: x[0]), reverse=True), key=(lambda x: x[1]))
    sorted_set = [a for a in sorted_set if a[1] != 0]   # remove open strings
    if len(sorted_set) >= 4:
        combinations = itertools.combinations([1]*(len(sorted_set)-2) + [2,3,4], len(sorted_set))
    else:
        combinations = itertools.combinations(range(1, 5), len(sorted_set))
    for combi in combinations:
        dico = defaultdict(list)
        for i, f in enumerate(combi):
            dico[f].append(sorted_set[i])
        skip = False
        for v in dico.values():
            if len(v) == 1:
                continue
            else:
                fret = v[0][1]
                for pair in v:
                    if pair[1] != fret:
                        skip = True
                        break
        if not skip:
            out.append(dico)
    return out

def _anatomical_score(fingering: Dict[int, List[Tuple[int, int]]]) -> float:
    score = 0
    fingers = list(fingering.keys())
    m = len(fingers)
    if m == 1:
        if len(fingering[fingers[0]]) == 1:
            # all one note "chords" are playable
            return 1
    div = max(m**2-m, 1)
    for i in range(len(fingers)):
        for j in range(i+1, len(fingers)):
            d1, d2 = fingers[i], fingers[j]
            for k in range(len(fingering[d1])):
                for l in range(len(fingering[d2])):
                    s1, s2 = fingering[d1][k][0], fingering[d2][l][0]
                    f1, f2 = fingering[d1][k][1], fingering[d2][l][1]
                    x = finger_distance(s1, f1, s2, f2)
                    fscore = finger_score(x, fingers[i]-1, fingers[j]-1)
                    score += fscore
    return score/div


def anatomical_score(diagram: str) -> Tuple[float, Dict[int, Tuple[int, int]]]:
    string_fret_set = get_string_fret_set(diagram)
    only_open_strings = True
    for sf in string_fret_set:
        if sf[1] != 0:
            only_open_strings = False
            break
    if only_open_strings:
        return 1, None
    fingerings = get_possible_fingerings(string_fret_set)
    score = 0
    best_fingering = None
    for fingering in fingerings:
        fscore = _anatomical_score(fingering)
        if fscore > score:
            score = fscore
            best_fingering = fingering
    return score, best_fingering






def ratio_muted_strings(diagram: str, num_strings: int = NUM_STRINGS) -> float:
    muted_strings = diagram.count('x')
    return muted_strings / num_strings

def ratio_open_strings(diagram: str, num_strings: int = NUM_STRINGS) -> float:
    open_strings = diagram.count('0')
    strings_played = num_strings_played(diagram, num_strings)
    if strings_played == 0:
        return 0
    return open_strings / strings_played

def num_strings_played(diagram: str, num_strings: int = NUM_STRINGS) -> float:
    muted_strings = diagram.count('x')
    return num_strings - muted_strings

def string_centroid(diagram: str, num_strings: int = NUM_STRINGS) -> float:
    centroid = 0
    string = num_strings
    for v in diagram.split('.'):
        if v in ['x', '']:
            string -= 1
            continue
        else:
            centroid += string
            string -= 1
    strings_played = num_strings_played(diagram, num_strings)
    if strings_played == 0:
        return 0
    centroid = centroid / strings_played
    return centroid


