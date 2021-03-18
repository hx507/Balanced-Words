# Convert non-binary automaton accepting alphabet of size n to accept binary inputs separated by 2
import numpy as np
import os
from copy import deepcopy
import sys
import pprint
pprint = pprint.PrettyPrinter(depth=6).pprint

aut_template = {
    'name': None,
    'alphabet': set(),
    'states': set(),
    'initial_state': None,
    'accepting_states': set(),
    'transitions': dict(),
}


# Parse an input generated from https://github.com/ReedOei/OstrowskiAutomata
def parse(filename, n):
    aut_to_read = deepcopy(aut_template)
    aut_to_read['alphabet'] = set(np.arange(n))

    with open(filename, encoding='utf-16-be') as f:
        # remove empty lines, comments, etc
        lines = list(filter(lambda x: x.replace(
            ' ', '').replace('\n', '').__len__() != 0,
            map(lambda x: x.split('//')[0].replace('\n', ''),
                f.readlines())))
        aut_to_read['name'], lines = lines[0], lines[1:]
        curr_state = None
        for l in lines:
            if '->' not in l:
                state, acc = list(map(lambda x: int(x), l.split(' ')))
                if curr_state is None:
                    aut_to_read['initial_state'] = curr_state
                if acc:
                    aut_to_read['accepting_states'].add(state)
                aut_to_read['states'].add(state)
                aut_to_read['transitions'][state] = dict()
                curr_state = state
            else:
                inp, to = list(map(lambda x: int(x), l.split('->')))
                aut_to_read['transitions'][curr_state][inp] = to
    return aut_to_read

# Generate a separate automata occupying states numbered between start_num~start_num+log2(num)+1,
# that would recognize [lsd_bin(num)]0^ω2 as input
# There is an optional final state to make concatenating more convenient


def creat_recognizing(num, start_num=0, final_state=None, start_state=None):
    aut_to_read = deepcopy(aut_template)
    aut_to_read['name'] = f'recg{num}'
    aut_to_read['alphabet'] = {0, 1, 2}

    nums_encoded = list(reversed(bin(num)))[:-2]
    # remove tailing zeros
    while len(nums_encoded) and nums_encoded[-1] == 0:
        nums_encoded = nums_encoded[:-1]
    # additional states: start, 0^ω, end
    aut_to_read['states'] = start_num + np.arange(len(nums_encoded)+2)
    if final_state is not None:
        aut_to_read['states'][-1] = final_state
    if start_state is not None:
        aut_to_read['states'][0] = start_state
    aut_to_read['initial_state'] = 0
    aut_to_read['accepting_states'].add(aut_to_read['states'][-1])

    for i in range(len(nums_encoded)):
        if i == len(nums_encoded)-1:  # last one
            aut_to_read['transitions'][aut_to_read['states'][i]] = {
                2: aut_to_read['states'][-1],
                nums_encoded[i]: aut_to_read['states'][i+1]}
        else:
            aut_to_read['transitions'][aut_to_read['states'][i]] = {
                nums_encoded[i]: aut_to_read['states'][i+1]}
    aut_to_read['transitions'][aut_to_read['states'][-2]] = {
        0: aut_to_read['states'][-2],
        2: aut_to_read['states'][-1]
    }
    return aut_to_read


def convert(filename, n):
    org_aut = parse(filename, n)
    print('Read original automaton:')
    pprint(org_aut)

    unoccupied_state_number = max(org_aut['states'])+100

    # reg1234 = creat_recognizing(1234, start_num=100)
    # print('Create automaton recognizing bin(1234)[2]:')
    # pprint(reg1234)

    for src_state in org_aut['transitions'].keys():
        for inp in org_aut['transitions'][src_state].keys():
            dst_state = org_aut['transitions'][src_state][inp]
            print(f'Bridging {src_state}-{inp}->{dst_state}:')

            bridge_aut = creat_recognizing(
                inp, start_num=unoccupied_state_number, final_state=dst_state, start_state=src_state)
            pprint(bridge_aut)

            first_transition = bridge_aut['transitions'][src_state]
            del bridge_aut['transitions'][src_state]

            print("first transition: ")
            pprint(first_transition)
            org_aut['states'] |= set(bridge_aut['states'])
            org_aut['transitions'].update(bridge_aut['transitions'])
            break
        break
    pass

    print("Final bridged automaton:")
    pprint(org_aut)


convert('./words/X5_0.txt', 5)
