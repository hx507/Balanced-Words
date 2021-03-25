# Convert non-binary automaton accepting alphabet of size n to accept binary inputs separated by 2
from multiprocessing import Process
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
                    aut_to_read['initial_state'] = state
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
    print(representations)

    transitions = [(start_state, 2, final_states[0])]
    curr_state = start_num
    for i, rep in enumerate(representations[1:]):
        transitions += [(start_state, rep[0], curr_state)]
        for inp in rep[1:]:
            transitions += [(curr_state, inp, curr_state+1)]
            curr_state += 1
        transitions += [(curr_state, 2, final_states[i+1])]
        curr_state += 1
        print(transitions)

    for t in transitions:
        aut['states'] |= {t[0], t[2]}
    for s in aut['states']:
        aut['transitions'][s] = {0: set(map(lambda x: x[2], filter(lambda x: x[0] == s and x[1] == 0, transitions))),
                                 1: set(map(lambda x: x[2], filter(lambda x: x[0] == s and x[1] == 1, transitions))),
                                 2: set(map(lambda x: x[2], filter(lambda x: x[0] == s and x[1] == 2, transitions))), }
    # remove empty sets
    aut['transitions'] = {
        src: {inp: v for inp, v in dst.items() if len(v)} for src, dst in aut['transitions'].items()}
    aut['transitions'] = {
        src: dst for src, dst in aut['transitions'].items() if len(dst)}

    print(f'Bridge from {start_state} -> {final_states}')
    pprint(aut)
    return aut


def convert(filename, n):
    # Parse Automaton
    org_aut = parse(filename, n)
    print(f'Read original automaton with alphabet of size {n}:')
    pprint(org_aut)

    unoccupied_state_number = max(org_aut['states'])+1

    # reg1234 = creat_recognizing(
    # n, start_num=10, start_state=-1, final_states=np.arange(n)+100)
    # print('Create automaton recognizing bin(1234)[2]:')
    # pprint(reg1234)

    if True:
        org_keys = list(org_aut['transitions'].keys())
        for src_state in org_keys:
            final_states = np.array([list(org_aut['transitions'][src_state][i])[
                0] if i in org_aut['transitions'][src_state] else -1 for i in range(n)])
            bridge_aut = creat_recognizing(
                n, start_num=unoccupied_state_number, final_states=final_states, start_state=src_state)
            # pprint(bridge_aut)

            org_aut['states'] |= set(bridge_aut['states'])
            org_aut['transitions'].update(bridge_aut['transitions'])
            unoccupied_state_number = max(org_aut['states'])+1
            # break

         # must start with 2
        new_start = max(org_aut['states'])+1
        org_aut['states'] |= {new_start, -1}
        org_aut['transitions'][new_start] = {2: {org_aut['initial_state']}}
        org_aut['initial_state'] = new_start
        print("Final bridged automaton:")
        pprint(org_aut)

    # Dump everything
    if True:
        out_string = ''
        out_string += ('{0,1,2}\n')
        all_states = sorted(list(org_aut['states']))
        all_states.remove(org_aut['initial_state'])
        all_states = [org_aut['initial_state']]+all_states
        print(all_states)
        for state in all_states:
            out_string += (
                f's{state}: {1 if state in org_aut["accepting_states"] else 0}\n')
            if state in org_aut['transitions'].keys():
                for inp in org_aut['transitions'][state].keys():
                    dst = "\n".join(
                        map(lambda x: f'{inp} -> s'+str(x), org_aut["transitions"][state][inp]))
                    out_string += dst+'\n'
        print(out_string)
        with open('./words_for_Pecan/'+filename.split('/')[-1], 'w') as out_file:
            out_file.write(out_string)


representation_alphabet_sizes = {
    3: 3, 4: 3, 5: 3, 6: 3, 7: 4, 8: 4, 9: 4, 10: 5}
convert(f'./words/X5_0.txt', representation_alphabet_sizes[5])
if True:
    processes = []
    for k in range(3, 11):
        for i in range(k):
            p = Process(target=convert, args=(
                f'./words/X{k}_{i}.txt', representation_alphabet_sizes[k]))
            p.start()
            processes += [p]
    for p in processes:
        p.join()
