# Convert non-binary automaton accepting alphabet of size n to accept binary inputs separated by 2
import numpy as np
import os
from copy import deepcopy
import sys
import pprint
pprint = pprint.PrettyPrinter(depth=5).pprint

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
                aut_to_read['transitions'][curr_state][inp] = set({to})
    return aut_to_read


def lsd_bin(num):
    nums_encoded = list(map(lambda x: int(x), reversed(bin(num)[2:])))
    # nums_encoded = list(map(lambda x: (x), reversed(bin(num)[2:])))
    while len(nums_encoded) and nums_encoded[-1] == 0:
        nums_encoded = nums_encoded[:-1]
    return nums_encoded

# Generate a separate automata occupying states numbered between start_num~start_num+log2(num)+1,
# that would recognize [lsd_bin(i)]0^ω2 for i=0..n as input


def creat_recognizing(n, start_num=0, final_states=None, start_state=0):
    aut = deepcopy(aut_template)
    aut['name'] = f'recg{n}'
    aut['alphabet'] = {0, 1, 2}

    representations = list(map(lsd_bin, range(n)))
    nlongest = max(map(len, representations))

    aut['states'] = np.arange(n*(nlongest+2)+1)+start_num

    aut['states'][-(n+1)] = start_state
    if final_states is not None:
        final_states[final_states == -
                     1] = aut['states'][-n:][final_states == -1]
        aut['states'][-n:] = final_states

    for s in aut['states'][:-n]:
        aut['transitions'][s] = {0: set(), 1: set(), 2: set()}
    # i in 0..n
    for i in range(n):
        rep = representations[i]
        for k in range(len(rep)):
            st = k*n+i
            inp = rep[k]
            end = (k+1)*n+i
            aut['transitions'][aut['states'][st]][inp].add(aut['states'][end])
            # if k == len(rep)-1:  # last
                # aut['transitions'][aut['states'][st]][2].add(final_states[i])
        aut['transitions'][aut['states']
                           [len(rep)*n+i]][0].add(aut['states'][len(rep)*n+i])
        aut['transitions'][aut['states']
                           [len(rep)*n+i]][2].add(final_states[i])

    aut['transitions'][start_state][0] = set.union(*[dst[0] for src,
                                                     dst in aut['transitions'].items() if src in aut['states'][:n]])
    aut['transitions'][start_state][1] = set.union(*[dst[1] for src,
                                                     dst in aut['transitions'].items() if src in aut['states'][:n]])
    aut['transitions'][start_state][2] = set.union(*[dst[2] for src,
                                                     dst in aut['transitions'].items() if src in aut['states'][:n]])
    # remove empty sets
    # aut['states'] = list(filter(lambda x: x in aut['transitions'] and len(
    # aut['transitions'][x].items()), aut['states']))
    aut['transitions'] = {
        src: {inp: v for inp, v in dst.items() if len(v)} for src, dst in aut['transitions'].items()}
    return aut


def convert(filename, n):
    org_aut = parse(filename, n)
    print('Read original automaton:')
    pprint(org_aut)

    unoccupied_state_number = max(org_aut['states'])+1

    # reg1234 = creat_recognizing(
    # n, start_num=10000, start_state=-1, final_states=np.arange(n)+10)
    # print('Create automaton recognizing bin(1234)[2]:')
    # pprint(reg1234)

    org_keys = list(org_aut['transitions'].keys())
    for src_state in org_keys:
        final_states = np.array([list(org_aut['transitions'][src_state][i])[
            0] if i in org_aut['transitions'][src_state] else -1 for i in range(n)])
        bridge_aut = creat_recognizing(
            n, start_num=unoccupied_state_number, final_states=final_states, start_state=src_state)
        pprint(bridge_aut)

        org_aut['states'] |= set(bridge_aut['states'])
        org_aut['transitions'].update(bridge_aut['transitions'])
        unoccupied_state_number = max(org_aut['states'])+1
        # break

    print("Final bridged automaton:")
    pprint(org_aut)


convert('./words/X5_0.txt', 5)
