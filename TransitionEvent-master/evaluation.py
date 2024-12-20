


import numpy as np
import dynet as dy
import nn
import ops
from dy_utils import ParamManager as pm
from actions import Actions
from vocab import Vocab
# from event_constraints import EventConstraint
import io_utils
from io_utils import read_yaml
from io_utils import read_pickle, write_lines, read_lines
import json

class EventConstraint(object):
    '''
       This class is used to make sure that (event types, entity types) -> (argument roles) obey event constraints. 
    '''

    def __init__(self, ent_dict, tri_dict, arg_dict):
        constraint_file = './data_files/argrole_dict.txt'

        self.constraint_list = [] # [(ent_type, tri_type, arg_type)]
        for line in read_lines(constraint_file):
            line = str(line).lower()
            arr = line.split()
            arg_type = arr[0]
            for pair in arr[1:]:
                pair_arr = pair.split(',')
                tri_type = pair_arr[0]
                ent_type = pair_arr[1]
                ent_type = self._replace_ent(ent_type)
                self.constraint_list.append((ent_type, tri_type, arg_type))

        print('Event constraint size:',len(self.constraint_list))
        # { (ent_type, tri_type) : (arg_type1, ...)}
        self.ent_tri_to_arg_hash = {}
        for cons in self.constraint_list:
            ent_id = ent_dict[cons[0]]
            tri_id = tri_dict[cons[1]]
            arg_id = arg_dict[cons[2]]
            # ent_id = cons[0]
            # tri_id = cons[1]
            # arg_id = cons[2]
            if (ent_id, tri_id) not in self.ent_tri_to_arg_hash:
                self.ent_tri_to_arg_hash[(ent_id, tri_id)] = set()

            self.ent_tri_to_arg_hash[(ent_id, tri_id)].add(arg_id)

        #print(self.ent_tri_to_arg_hash)

        # single = 0
        # for key, val in self.ent_tri_to_arg_hash.items():
        #     if len(val) == 1:
        #         single += 1
        # print(single)

    def _replace_ent(self, ent_type):
        if ent_type == 'time':
            return 'tim'

        if ent_type == 'value':
            return 'val'

        return ent_type

    def check_constraint(self, ent_type, tri_type, arg_type):
        if (ent_type, tri_type, arg_type) in self.constraint_list:
            return True
        else:
            return False


    def get_constraint_arg_types(self, ent_type_id, tri_type_id):
        return self.ent_tri_to_arg_hash.get((ent_type_id, tri_type_id), None)


class LambdaVar(object):
    TRIGGER = 't'
    ENTITY = 'e'
    OTHERS = 'o'

    def __init__(self, ):
        # self.var = None
        self.idx = -1
        # self.lambda_type = LambdaVar.OTHERS

    def push(self,  idx):
        self.idx = idx
        # self.lambda_type = lambda_type

    def pop(self):
        idx =  self.idx
        self.idx = -1
        # self.lambda_type = LambdaVar.OTHERS
        return idx

    # def clear(self):
    #     self.var, self.idx = None, -1

    # def is_empty(self):
    #     return self.var is None

    # def is_trigger(self):
    #     return self.lambda_type == LambdaVar.TRIGGER

    # def is_entity(self):
    #     return self.lambda_type == LambdaVar.ENTITY

class Buffer(object):

    def __init__(self,  hidden_state_list):
        '''

        :param state_tensor: list of (n_dim)
        '''

        self.hidden_states = hidden_state_list
        print(hidden_state_list)
        self.seq_len = len(hidden_state_list)
        self.idx = 0

    def pop(self):
        if self.idx == self.seq_len:
            raise RuntimeError('Empty buffer')
        hx = self.hidden_states[self.idx]
        cur_idx = self.idx
        self.idx += 1
        return cur_idx

    def last_state(self):
        return  self.idx

    def buffer_idx(self):
        return self.idx

    def is_empty(self):
        return (self.seq_len - self.idx) == 0

    def move_pointer(self, idx):
        self.idx = idx

    def move_back(self):
        self.idx -= 1

    def __len__(self):
        return  self.seq_len - self.idx

class ShiftReduce(object):

    def __init__(self, action_dict, ent_dict, tri_dict, arg_dict):
        self.sigma_rnn = []
        self.delta_rnn = []

        self.part_ent_rnn = []
        self.out_rnn = []
        self.lambda_var = LambdaVar()
        
        self.trigger_gen = 'TRIGGER-GEN-'

        self.entity_shift = 'ENTITY-SHIFT'
        self.entity_gen = 'ENTITY-GEN-'
        self.entity_back = 'ENTITY-BACK'

        self.o_delete = 'O-DELETE'


        self.event_gen = 'EVENT-GEN-'

        self.shift = 'SHIFT'
        self.no_pass = 'NO-PASS'
        self.left_pass = 'LEFT-PASS'
        self.right_pass = 'RIGHT-PASS'
        self.left_relate = 'LEFT-RELATE'
        self.right_relate = 'RIGHT-RELATE'

        self.back_shift = 'DUAL-SHIFT'
        # ------------------------------
        self.copy_shift = 'COPY-SHIFT'


        # ------------------------------
        # event_shift = 'EVENT-SHIFT-'
        # event_reduce = 'EVENT-REDUCE-'
        # no_reduce = 'NO-REDUCE'

        self._PASS_PLACEHOLDER = 'PASS'
        self._SHIFT_PLACEHOLDER = 'SHIFT'

        # self.act_to_ent_id = {}
        # self.act_to_tri_id = {}
        # self.act_to_arg_id = {}
        # self.arg_to_act_id = {}

        # self.ent_gen_group = set()
        # self.tri_gen_group = set()
        # self.event_gen_group = set()

        # self.act_id_to_str = {v:k for k, v in action_dict.items()}

        # for name, id in action_dict.items():
        #     if name.startswith(self.entity_gen):
        #         self.ent_gen_group.add(id)
        #         self.act_to_ent_id[name] = ent_dict[name[len(self.entity_gen):]]

        #     elif name.startswith(self.trigger_gen):
        #         self.tri_gen_group.add(id)
        #         self.act_to_tri_id[name] = tri_dict[name[len(self.trigger_gen):]]

        #     elif name.startswith(self.event_gen):
        #         self.event_gen_group.add(id)
        #         self.act_to_arg_id[name] = arg_dict[name[len(self.event_gen):]]

        # for k,v in self.act_to_arg_id.items():
        #     self.arg_to_act_id[v] = k
        
        # self.event_cons = EventConstraint(ent_dict, tri_dict, arg_dict)

    def __call__(self, actions, sentence):
        ent_dic = dict()
        tri_dic = dict()

        args = []
        buffer = Buffer(sentence)
        pred_action_strs = []
        relations = []

        for action in actions:
            # print(ent_dic, tri_dic, args, relations, buffer)
            if action == self.o_delete:
                idx = buffer.pop()

            elif self.trigger_gen in action:
                idx = buffer.pop()
                type_id = action.split('-')[-1]
                tri_dic[idx] = (idx, type_id)
                self.lambda_var.push(idx)

            elif action == self.entity_shift:
                if not buffer:
                    break
                idx = buffer.pop()
                self.part_ent_rnn.append(idx)

            elif self.entity_gen in action:
                start, end = self.part_ent_rnn[0], self.part_ent_rnn[-1]
                type_id = action.split('-')[-1]
                ent = (start, end, type_id)
                ent_dic[start] = ent
                self.lambda_var.push(start)

            elif action == self.entity_back:
                new_idx = buffer.idx
                new_idx -= len(self.part_ent_rnn) - 1
                buffer.move_pointer(new_idx)

                self.part_ent_rnn = []


            elif action == self.shift:
                while self.delta_rnn:
                    self.sigma_rnn.append(self.delta_rnn.pop())
                self.sigma_rnn.append(self.lambda_var.pop())


            elif action == self.copy_shift:
                while not self.delta_rnn.is_empty():
                    self.sigma_rnn.append(self.delta_rnn.pop())

                self.sigma_rnn.append(self.lambda_var.pop())
                buffer.move_back()


            elif action == self.no_pass:
                lmda_idx = self.lambda_var.idx
                sigma_last_idx = self.sigma_rnn.pop()

                self.delta_rnn.append(sigma_last_idx)


            elif self.left_pass in action:
                lmda_idx = self.lambda_var.idx
                sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = lmda_idx
                ent_start, ent_end, _ = ent_dic[sigma_last_idx]

                role_label = action.split('-')[-1]
                event = (ent_start, ent_end, tri_idx, role_label)
                args.append(event)

                self.delta_rnn.append(sigma_last_idx)
            


            elif self.right_pass in action:
                lmda_idx = self.lambda_var.idx
                sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = sigma_last_idx
                ent_start, ent_end, _ = ent_dic[lmda_idx]

                role_label = action.split('-')[-1]

                event = (ent_start, ent_end, tri_idx, role_label)
                args.append(event)

                self.delta_rnn.append(sigma_last_idx)

            elif self.left_relate in action:
                lmda_idx = self.lambda_var.idx
                sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = lmda_idx
                ent_start, ent_end, _ = ent_dic[sigma_last_idx]
                ent_start_, ent_end_, _ = ent_dic[tri_idx]

                role_label = action.split('-')[-1]
                relation = (ent_start, ent_end, ent_start_, ent_end_, role_label)
                relations.append(relation)

                self.delta_rnn.append(sigma_last_idx)
            


            elif self.right_relate in action:
                lmda_idx = self.lambda_var.idx
                sigma_last_idx = self.sigma_rnn.pop()
                tri_idx = sigma_last_idx
                ent_start, ent_end, _ = ent_dic[lmda_idx]
                ent_start_, ent_end_, _ = ent_dic[tri_idx]

                role_label = action.split('-')[-1]

                relation = (ent_start, ent_end, ent_start_, ent_end_, role_label)
                relations.append(relation)

                self.delta_rnn.append(sigma_last_idx)

            else:
                raise RuntimeError('Unknown action type:'+str(action))

        pred_args = args
        pred_relations = relations
        # for arg in args:
        #     ent_start, ent_end, tri_idx, role_type = arg
        #     ent_type_id = ent_dic[ent_start][-1]
        #     tri_type_id = tri_dic[tri_idx][-1]
        #     valid_args = self.event_cons.get_constraint_arg_types(ent_type_id, tri_type_id)
        #     if valid_args and role_type in valid_args:
        #         pred_args.append(arg)

        return set(ent_dic.values()), set(tri_dic.values()), pred_args, pred_relations

class evaluater(object):
    def __init__(self):
        self.train_list, self.dev_list, self.test_list, self.token_vocab, self.char_vocab, self.ent_type_vocab, \
            self.ent_ref_vocab, self.tri_type_vocab, self.arg_type_vocab, self.action_vocab, \
            self.pos_vocab = read_pickle(data_config['inst_pl_file'])
        self.shift_reduce = ShiftReduce(
                                        self.action_vocab,
                                        self.ent_type_vocab,
                                        self.tri_type_vocab,
                                        self.arg_type_vocab,
                                        )

if __name__ == '__main__':
    data_config = read_yaml('data_config.yaml')
    evaluation = evaluater()
    states = [
    "ENTITY-SHIFT",
    "ENTITY-GEN-PER",
    "SHIFT",
    "ENTITY-BACK",
    "O-DELETE",
    "TRIGGER-GEN-Sentence",
    "LEFT-PASS-Defendant",
    "SHIFT",
    "O-DELETE",
    "ENTITY-SHIFT",
    "ENTITY-SHIFT",
    "ENTITY-SHIFT",
    "ENTITY-SHIFT",
    "ENTITY-GEN-VAL",
    "RIGHT-PASS-Sentence",
    "NO-PASS",
    "SHIFT",
    "ENTITY-BACK",
    "O-DELETE",
    "O-DELETE",
    "ENTITY-SHIFT",
    "ENTITY-GEN-FAC"
    ]
    # states = ["O-DELETE", "O-DELETE", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-GEN-G#protein", "ENTITY-BACK", "SHIFT", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "O-DELETE", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-GEN-G#DNA", "ENTITY-BACK", "NO-PASS", "SHIFT", "O-DELETE", "O-DELETE", "O-DELETE", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-GEN-G#DNA", "ENTITY-BACK", "NO-PASS", "NO-PASS", "SHIFT", "O-DELETE", "O-DELETE", "O-DELETE", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-SHIFT", "ENTITY-GEN-G#DNA", "ENTITY-BACK", "NO-PASS", "NO-PASS", "NO-PASS", "SHIFT", "O-DELETE", "O-DELETE", "O-DELETE"]
    # role_labels = ['Defendant', 'Sentence']
    # sentence= "LOS ANGELES , April 18 ( AFP )"
    # states= "{\"actions\": [\"ENTITY-SHIFT\", \"ENTITY-SHIFT\", \"ENTITY-GEN-GPE\", \"ENTITY-BACK\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-ORG\", \"ENTITY-BACK\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\"]}"
    sentence = "Best - selling novelist and \" Jurassic Park \" creator Michael Crichton has agreed to pay his fourth wife 31 million dollars as part of their divorce settlement , court documents showed Friday ."
    states = "{\"actions\": [\"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"NO-PASS\", \"SHIFT\", \"ENTITY-SHIFT\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"TRIGGER-GEN-Transaction:Transfer-Money\", \"LEFT-PASS-Giver\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"RIGHT-RELATE-PER-SOC:Family\", \"RIGHT-PASS-Recipient\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-PER\", \"ENTITY-BACK\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"TRIGGER-GEN-Life:Divorce\", \"LEFT-PASS-Person\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"ENTITY-SHIFT\", \"ENTITY-GEN-ORG\", \"ENTITY-BACK\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"NO-PASS\", \"SHIFT\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\", \"O-DELETE\"]}"
    entities, triggers, args, relations  = evaluation.shift_reduce(json.loads(states)['actions'], sentence)
    print(entities, triggers, args, relations)
    '''
    {"prompt": "[INST] Would you be able to identify the two entities described in the text and specify their connection?\n{\"ent\": [\"opinion\", \"aspect\"], \"rel\": [\"negative\", \"neutral\", \"positive\"], \"event\": {}}\nBoot time is super fast , around anywhere from 35 seconds to 1 minute . [/INST]", 
    "label": "{\"ent\": [{\"type\": \"aspect\", \"text\": \"Boot time\", \"span\": [0, 9]}, {\"type\": \"opinion\", \"text\": \"fast\", \"span\": [18, 22]}], \"rel\": [{\"relation\": \"positive\", \"head\": {\"text\": \"Boot time\", \"span\": [0, 9]}, \"tail\": {\"text\": \"fast\", \"span\": [18, 22]}}], \"event\": []}", 
    "predict": "{\"ent\": [{\"type\": \"aspect\", \"text\": \"Boot time\", \"span\": [0, 9]}, {\"type\": \"opinion\", \"text\": \"super fast\", \"span\": [12, 22]}], \"rel\": [{\"relation\": \"positive\", \"head\": {\"text\": \"Boot time\", \"span\": [0, 9]}, \"tail\": {\"text\": \"super fast\", \"span\": [12, 22]}}], \"event\": []}"}
    {"prompt": "[INST] Please retrieve relevant details about the event mentioned in the given text, including the trigger, type, and related components.\n{\"ent\": [\"geopolitical entity\", \"time\", \"file\", \"website\", \"data\", \"common vulnerabilities and exposures\", \"money\", \"patch\", \"malware\", \"person\", \"purpose\", \"number\", \"vulnerability\", \"version\", \"capabilities\", \"payment method\", \"system\", \"software\", \"personally identifiable information\", \"organization\", \"device\"], \"rel\": [], \"event\": {\"phishing\": [\"purpose\", \"time\", \"victim\", \"trusted entity\", \"place\", \"attack pattern\", \"attacker\", \"damage amount\", \"tool\"], \"databreach\": [\"time\", \"purpose\", \"compromised data\", \"number of victim\", \"victim\", \"number of data\", \"place\", \"attack pattern\", \"attacker\", \"damage amount\", \"tool\"], \"ransom\": [\"time\", \"victim\", \"price\", \"place\", \"attack pattern\", \"attacker\", \"payment method\", \"damage amount\", \"tool\"], \"discover vulnerability\": [\"time\", \"discoverer\", \"vulnerable system version\", \"supported platform\", \"common vulnerabilities and exposures\", \"vulnerability\", \"capabilities\", \"vulnerable system\", \"vulnerable system owner\"], \"patch vulnerability\": [\"time\", \"releaser\", \"patch number\", \"vulnerable system version\", \"common vulnerabilities and exposures\", \"supported platform\", \"issues addressed\", \"patch\", \"vulnerability\", \"vulnerable system\"]}}\nIn describing a phishing attack , UConn Health says that on Dec 24 , 2018 , it determined that an unauthorized third party illegally accessed a limited number of employee email accounts containing patient information , including some individuals ' names , dates of birth , addresses and limited medical information , such as billing and appointment information . [/INST]", 
    "label": "{\"ent\": [{\"type\": \"organization\", \"text\": \"UConn Health\", \"span\": [33, 45]}, {\"type\": \"time\", \"text\": \"Dec 24 , 2018\", \"span\": [59, 72]}, {\"type\": \"organization\", \"text\": \"unauthorized third party\", \"span\": [97, 121]}, {\"type\": \"person\", \"text\": \"individuals\", \"span\": [233, 244]}, {\"type\": \"personally identifiable information\", \"text\": \"employee email accounts\", \"span\": [161, 184]}, {\"type\": \"personally identifiable information\", \"text\": \"patient information\", \"span\": [196, 215]}, {\"type\": \"personally identifiable information\", \"text\": \"names\", \"span\": [247, 252]}, {\"type\": \"personally identifiable information\", \"text\": \"dates of birth\", \"span\": [255, 269]}, {\"type\": \"personally identifiable information\", \"text\": \"addresses\", \"span\": [272, 281]}, {\"type\": \"data\", \"text\": \"limited medical information\", \"span\": [286, 313]}, {\"type\": \"data\", \"text\": \"billing\", \"span\": [324, 331]}, {\"type\": \"data\", \"text\": \"appointment information\", \"span\": [336, 359]}], \"rel\": [], 
    \"event\": [{\"event_type\": \"phishing\", \"trigger\": {\"text\": \"phishing attack\", \"span\": [15, 30]}, 
    \"args\": [{\"role\": \"victim\", \"text\": \"UConn Health\", \"span\": [33, 45]}, {\"role\": \"time\", \"text\": \"Dec 24 , 2018\", \"span\": [59, 72]}]}, {\"event_type\": \"databreach\", \"trigger\": {\"text\": \"illegally accessed\", \"span\": [122, 140]}, \"args\": [{\"role\": \"attacker\", \"text\": \"unauthorized third party\", \"span\": [97, 121]}, {\"role\": \"victim\", \"text\": \"individuals\", \"span\": [233, 244]}, {\"role\": \"compromised data\", \"text\": \"employee email accounts\", \"span\": [161, 184]}, {\"role\": \"compromised data\", \"text\": \"patient information\", \"span\": [196, 215]}, {\"role\": \"compromised data\", \"text\": \"names\", \"span\": [247, 252]}, {\"role\": \"compromised data\", \"text\": \"dates of birth\", \"span\": [255, 269]}, {\"role\": \"compromised data\", \"text\": \"addresses\", \"span\": [272, 281]}, {\"role\": \"compromised data\", \"text\": \"limited medical information\", \"span\": [286, 313]}, {\"role\": \"compromised data\", \"text\": \"billing\", \"span\": [324, 331]}, {\"role\": \"compromised data\", \"text\": \"appointment information\", \"span\": [336, 359]}]}]}", 
    "predict": "{\"ent\": [{\"type\": \"organization\", \"text\": \"UConn Health\", \"span\": [32, 44]}, {\"type\": \"time\", \"text\": \"Dec 24 , 2018\", \"span\": [58, 71]}, {\"type\": \"personally identifiable information\", \"text\": \"employee email accounts\", \"span\": [161, 185]}, {\"type\": \"personally identifiable information\", \"text\": \"patient information\", \"span\": [191, 209]}, {\"type\": \"personally identifiable information\", \"text\": \"billing and appointment information\", \"span\": [326, 366]}, {\"type\": \"personally identifiable information\", \"text\": \"appointment information\", \"span\": [370, 394]}, {\"type\": \"person\", \"text\": \"individuals\", \"span\": [241, 252]}, {\"type\": \"person\", \"text\": \"some individuals\", \"span\": [255, 271]}], \"rel\": [], \"event\": [{\"event_type\": \"databreach\", \"trigger\": {\"text\": \"illegally accessed\", \"span\": [122, 139]}, \"args\": [{\"role\": \"victim\", \"text\": \"UConn Health\", \"span\": [32, 44]}, {\"role\": \"time\", \"text\": \"Dec 24 , 2018\", \"span\": [58, 71]}, {\"role\": \"compromised data\", \"text\": \"employee email accounts\", \"span\": [161, 185]}, {\"role\": \"compromised data\", \"text\": \"patient information\", \"span\": [191, 209]}, {\"role\": \"compromised data\", \"text\": \"billing and appointment information\", \"span\": [326, 366]}, {\"role\": \"compromised data\", \"text\": \"appointment information\", \"span\": [370, 394]}, {\"role\": \"victim\", \"text\": \"individuals\", \"span\": [241, 252]}, {\"role\": \"victim\", \"text\": \"some individuals\", \"span\": [255, 271]}]}]}"}

    '''
    # # print(triggers, args)
    # # paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2/ace_add_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2/ace2004_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2/casie_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2/conll03_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2/genia_test/generated_predictions.jsonl']
    # # paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.2/ace_add_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.2/ace2004_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.2/casie_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.2/conll03_test/generated_predictions.jsonl', \
    # #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.2/genia_test/generated_predictions.jsonl']
    # paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.3/ace_add_test/generated_predictions.jsonl', \
    #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.3/ace2004_test/generated_predictions.jsonl', \
    #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.3/casie_test/generated_predictions.jsonl', \
    #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.3/conll03_test/generated_predictions.jsonl', \
    #     '/data/liuweichang/workspace/LLaMA-Factory-main/saves/llama2-7b/lora/predict_v2.3/genia_test/generated_predictions.jsonl']
    # for path in paths:
    #     with open(path, 'r', encoding='utf-8') as f:
    #         lines = f.readlines()
    #         all_datas = []
    #         for line in lines:
    #             line = json.loads(line)
    #             all_data = dict()
    #             all_data['prompt'] = line['prompt']
    #             all_data['actions'] = json.loads(line['label'])['actions']
    #             all_data['predict_actions'] = line['predict']
    #             all_data['predict'] = dict()
    #             all_data['label'] = dict()
    #             all_data['label']['ent'] = []
    #             all_data['label']['rel'] = []
    #             all_data['label']['event'] = []
    #             all_data['predict']['ent'] = []
    #             all_data['predict']['rel'] = []
    #             all_data['predict']['event'] = []
    #             sentence = line['prompt'].split('\n\n\n')[-1].replace(' [/INST]', '')

    #             entities, triggers, args = evaluation.shift_reduce(json.loads(line['label'])['actions'], sentence)
    #             for entity in entities:
    #                 all_data['label']['ent'].append({'type':entity[2], 'text':' '.join(sentence.split()[entity[0]:(entity[1]+1)]), 'span':[str(entity[0]), str(entity[1])]})
    #             for trigger in triggers:
    #                 tmp = dict()
    #                 tmp['event_type'] = trigger[1]
    #                 tmp['trigger'] = {'text':sentence.split()[trigger[0]], 'span':str(trigger[0])}
    #                 tmp['args'] = []
    #                 for arg in args:
    #                     if trigger[0] == arg[2]:
    #                         tmp['args'].append({'role':arg[-1], 'text':' '.join(sentence.split()[arg[0]:(arg[1] + 1)]), 'span':[str(arg[0]), str(arg[1])]})
    #                 all_data['label']['event'].append(tmp)
                
    #             try:
    #                 entities_pre, triggers_pre, args_pre = evaluation.shift_reduce(json.loads(line['predict'])['actions'], sentence)
    #             except Exception as e:
    #                 print(e)
    #                 entities_pre, triggers_pre, args_pre =[], [], []
    #             for entity in entities_pre:
    #                 all_data['predict']['ent'].append({'type':entity[2], 'text':' '.join(sentence.split()[entity[0]:(entity[1]+1)]), 'span':[str(entity[0]), str(entity[1])]})
    #             for trigger in triggers_pre:
    #                 if trigger[0] >= len(sentence.split()):
    #                     continue
    #                 tmp = dict()
    #                 tmp['event_type'] = trigger[1]
    #                 tmp['trigger'] = {'text':sentence.split()[trigger[0]], 'span':str(trigger[0])}
    #                 tmp['args'] = []
    #                 for arg in args_pre:
    #                     if trigger[0] == arg[2]:
    #                         tmp['args'].append({'role':arg[-1], 'text':' '.join(sentence.split()[arg[0]:(arg[1] + 1)]), 'span':[str(arg[0]), str(arg[1])]})
    #                 all_data['predict']['event'].append(tmp)
    #             all_data['label'] = json.dumps(all_data['label'], ensure_ascii=False)
    #             all_data['predict'] = json.dumps(all_data['predict'], ensure_ascii=False)
    #             all_datas.append(all_data)
    #             # print(all_data)
    #             # stop
    #     with open(path + 'propressed', 'w', encoding='utf-8') as f:
    #         for data in all_datas:
    #             f.write(json.dumps(data, ensure_ascii=False) + '\n')