# -*- coding: utf-8 -*-

'''
   Read data from JSON files,
   {"Sent": "he lost an election to a dead man .", "Triggers": [[3, "elect"]], "Entities": [[0, 0, "per", "E7"], [7, 7, "per", "E10"]], "Arguments": [[7, 7, 3, "Person"]], "nlp_words": ["He", "lost", "an", "election", "to", "a", "dead", "man", "."]}
'''
import os

import json
from collections import Counter

import numpy as np
import argparse

from io_utils import read_yaml, read_lines, read_json_lines, load_embedding_dict, save_pickle
from str_utils import capitalize_first_char, normalize_tok, normalize_sent, collapse_role_type
from vocab import Vocab

from actions import Actions

data_config = read_yaml('config.yaml')

parser = argparse.ArgumentParser(description = 'this is a description')
parser.add_argument('--seed', '-s', required = False, type = int, default=data_config['random_seed'])
args = parser.parse_args()
data_config['random_seed'] = args.seed
print('seed:',data_config['random_seed'])

np.random.seed(data_config['random_seed'])

data_dir = data_config['data_dir']
ace05_event_dir = data_config['ace05_event_dir']

def construct_instance(inst_list):
    word_num = 0
    processed_inst_list = []
    sample_sent_total = 2000
    sample_sent_num = 0
    for inst in inst_list:
        # print(inst)
        try:
            words = inst['nlp_words']
            tris = inst['Triggers'] # (idx, event_type)
            ents = inst['Entities'] # (start, end, coarse_type, ref_type)
            args = inst['Arguments'] # (ent_start, ent_end, trigger_idx, argument_type)
            relations = inst['Relations']
        except Exception as e:
            print(e)
            print(inst)
            stop

        # collapsed_args = []
        # for arg in args:
        #     collapsed_type = collapse_role_type(arg[3]).lower()
        #     collapsed_args.append([arg[0], arg[1], arg[2], collapsed_type])
        # inst['Arguments'] = collapsed_args

        actions = Actions.make_oracle(words, tris, ents, args, relations)
        inst['actions'] = actions

        processed_inst_list.append(inst)

    return processed_inst_list



def pickle_data():
    # paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition/train.txt', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition/test.txt', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition/dev.txt', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition/train_english.oneie.json', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition/test_english.oneie.json', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition/dev_english.oneie.json', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition/train.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition/test.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition/dev.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition/train.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition/test.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition/dev.jsonl', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition/train.data', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition/test.data', \
    # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition/dev.data']
    paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_2/train.txt', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_2/test.txt', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_2/dev.txt', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_2/train_english.oneie.json', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_2/test_english.oneie.json', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_2/dev_english.oneie.json', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_2/train.data', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_2/test.data', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_2/dev.data', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/transition_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/transition_2/dev.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/transition_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/transition_2/train.jsonl']
    for path in paths:
        data_list = read_json_lines(path)
        processed_data = construct_instance(data_list, )
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/transition_res_2/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/transition_res_2/')
        with open('/'.join(path.split('/')[:-2]) + '/transition_res_2/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
            for data in processed_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    # build_vocab()
    pickle_data()

