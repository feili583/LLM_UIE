import os
import json

'''
{"Sent": "he lost an election to a dead man .", "Triggers": [[3, "elect"]], "Entities": [[0, 0, "per", "E7"], [7, 7, "per", "E10"]], "Arguments": [[7, 7, 3, "Person"]], "nlp_words": ["He", "lost", "an", "election", "to", "a", "dead", "man", "."]}
'''

def find_str(text, substring):
    words = text.split(' ')

    substring_words = substring.split(' ')

    for i in range(len(words) - len(substring_words) + 1):
        # print(words[i:i + len(substring_words)], substring_words)
        if words[i:i + len(substring_words)] == substring_words:
            start_index = i
            return start_index, start_index + len(substring_words) -1 

def transfer2transition():
    def conll03():
        entity_formats = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/train.jsonl', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/test.jsonl', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/dev.jsonl']
        
        for entity_format in entity_formats:
            with open(entity_format, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_data = []
                for line in lines:
                    line = json.loads(line)
                    line['Sent'] = line['text']
                    line['Triggers'] = []
                    line['Entities'] = []
                    line['Arguments'] = []
                    line['Relations'] = []
                    line['nlp_words'] = line['text'].split()
                    for ent in line['ans']['ent']:                      
                        new_start, new_end = find_str(line['text'], ent['text'])
                        line['Entities'].append([new_start, new_end, ent['type']])
                    # stop
                    new_data.append(line)
            if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
            with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                    for data in new_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')

        # entity_formats = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/train.data', \
        # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/test.data', \
        # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/dev.data']
        
        # for entity_format in entity_formats:
        #     with open(entity_format, 'r', encoding='utf-8') as f:
        #         lines = f.readlines()
        #         new_data = []
        #         for id in range(0, len(lines), 4):
        #             sentence = lines[id].strip()
        #             tags = lines[id + 2].strip()
        #             line = dict()
        #             line['Sent'] = sentence
        #             line['Triggers'] = []
        #             line['Entities'] = []
        #             line['Arguments'] = []
        #             line['Relations'] = []
        #             line['nlp_words'] = sentence.split()
        #             for ent in tags.split('|'):
        #                 if ent:
        #                     index, tag = ent.split(' ')
        #                     new_start, new_end = index.split(',')
        #                     line['Entities'].append([int(new_start), int(new_end)-1, tag])
        #                 # else:
        #                 #     print(sentence)
        #             new_data.append(line)
        #     if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
        #         os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
        #     with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
        #             for data in new_data:
        #                 f.write(json.dumps(data, ensure_ascii=False) + '\n')
    def genia():
        entity_formats = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/train.data', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/test.data', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/dev.data', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/train.txt', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/test.txt', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/dev.txt']

        for entity_format in entity_formats:
            with open(entity_format, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_data = []
                for id in range(0, len(lines), 4):
                    sentence = lines[id].strip()
                    tags = lines[id + 2].strip()
                    line = dict()
                    line['Sent'] = sentence
                    line['Triggers'] = []
                    line['Entities'] = []
                    line['Arguments'] = []
                    line['Relations'] = []
                    line['nlp_words'] = sentence.split()
                    for ent in tags.split('|'):
                        if ent:
                            index, tag = ent.split(' ')
                            new_start, new_end = index.split(',')
                            line['Entities'].append([int(new_start), int(new_end)-1, tag])
                        # else:
                        #     print(sentence)
                    new_data.append(line)
            if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
            with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                    for data in new_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def casie():
        entity_formats = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/train.jsonl', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/test.jsonl', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/dev.jsonl']

        for entity_format in entity_formats:
            with open(entity_format, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_data = []
                for line in lines:
                    line = json.loads(line)
                    line['Sent'] = line['text']
                    line['Triggers'] = []
                    line['Entities'] = []
                    line['Arguments'] = []
                    line['Relations'] = []
                    line['nlp_words'] = line['text'].split()
                    for ent in line['ans']['ent']:
                        
                            
                        new_start, new_end = find_str(line['text'], ent['text'])
                        line['Entities'].append([new_start, new_end, ent['type']])
                    for event in line['ans']['event']:
                        trigger_start, trigger_end = find_str(line['text'], event['trigger']['text'])
                        line['Triggers'].append([trigger_end,  event['event_type']])
                        for args in event['args']:
                            new_start, new_end = find_str(line['text'], args['text'])
                            line['Arguments'].append([new_start, new_end, trigger_end, args['role']])

                    new_data.append(line)
            if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
            with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                    for data in new_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    def ace_add():
        entity_formats = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/train_english.oneie.json', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/test_english.oneie.json', \
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/dev_english.oneie.json']

        for entity_format in entity_formats:
            with open(entity_format, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                new_data = []
                for line in lines:
                    line = json.loads(line)
                    line['Sent'] = ' '.join(line['tokens'])
                    line['Triggers'] = []
                    line['Entities'] = []
                    line['Arguments'] = []
                    line['Relations'] = []
                    line['nlp_words'] = line['tokens']
                    for ent in line['entity_mentions']:                        
                        line['Entities'].append([ent['start'], ent['end']-1, ent['entity_type']])
                    for event in line['event_mentions']:
                        line['Triggers'].append([event['trigger']['end']-1,  event['event_type']])
                        for args in event['arguments']:
                            new_start, new_end = -1, -1
                            for new_start, new_end, lab in line['Entities']:
                                if ' '.join(line['tokens'][new_start:new_end+1]) == args['text']:
                                    line['Arguments'].append([new_start, new_end, event['trigger']['end']-1, args['role']])
                    for relation in line['relation_mentions']:
                        entities = []
                        for argu in relation['arguments']:
                            for entity in line['entity_mentions']:
                                if entity['id'] == argu['entity_id']:
                                    entities.append(entity['start'])
                                    entities.append(entity['end'] - 1)
                                    break
                        line['Relations'].append(entities + [relation['relation_subtype']])
                    new_data.append(line)
            if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
            with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                    for data in new_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')

    def rel():
        paths = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel'
        rels = ['conll04', 'nyt', 'scierc']
        files = ['train.jsonl', 'test.jsonl', 'dev.jsonl']
        for rel in rels:
            for file in files:
                # print(paths , rel , file)
                entity_format = paths + '/' + rel + '/' + file
                with open(entity_format, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    new_data = []
                    for line in lines:
                        line = json.loads(line)
                        line['Sent'] = line['text']
                        line['Triggers'] = []
                        line['Entities'] = []
                        line['Arguments'] = []
                        line['Relations'] = []
                        line['nlp_words'] = line['text'].split()
                        for ent in line['ans']['ent']:                            
                            new_start, new_end = find_str(line['text'], ent['text'])
                            line['Entities'].append([new_start, new_end, ent['type']])
                        for rela in line['ans']['rel']:                            
                            new_start1, new_end1 = find_str(line['text'], rela['head']['text'])
                            new_start2, new_end2 = find_str(line['text'], rela['tail']['text'])
                            line['Relations'].append([new_start1, new_end1, new_start2, new_end2, rela['relation']])
                        new_data.append(line)
                if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                    os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
                with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                        for data in new_data:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    def absa():
        paths = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa'
        rels = ['14lap', '14res', '15res', '16res']
        files = ['train.jsonl', 'test.jsonl', 'dev.jsonl']
        for rel in rels:
            for file in files:
                entity_format = paths + '/' + rel + '/' + file
                with open(entity_format, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    new_data = []
                    for line in lines:
                        line = json.loads(line)
                        line['Sent'] = line['text']
                        line['Triggers'] = []
                        line['Entities'] = []
                        line['Arguments'] = []
                        line['Relations'] = []
                        line['nlp_words'] = line['text'].split()
                        for ent in line['ans']['ent']:                            
                            new_start, new_end = find_str(line['text'], ent['text'])
                            line['Entities'].append([new_start, new_end, ent['type']])
                        for rela in line['ans']['rel']:                            
                            new_start1, new_end1 = find_str(line['text'], rela['head']['text'])
                            new_start2, new_end2 = find_str(line['text'], rela['tail']['text'])
                            line['Relations'].append([new_start1, new_end1, new_start2, new_end2, rela['relation']])
                        new_data.append(line)
                if not os.path.exists('/'.join(entity_format.split('/')[:-1]) + '/transition_2/'):
                    os.makedirs('/'.join(entity_format.split('/')[:-1]) + '/transition_2/')
                with open('/'.join(entity_format.split('/')[:-1]) + '/transition_2/' + entity_format.split('/')[-1], 'w', encoding='utf-8') as f:
                        for data in new_data:
                            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    conll03()
    genia()
    casie()
    ace_add()
    rel()
    absa()

def transfer2llama():
    instruction = "##Definitions:\n\n### 1. Transition states:\ns=(σ,δ,λ,e,β,T,E,R)\nWhere each symbol is defined as follows:\n- **σ**: A stack used to store processed elements. It holds the processed triggers and entities.\n- **δ**: A queue that stores elements temporarily popped from the stack, which will be pushed back into the stack in the future.\n- **λ**: A variable holding a reference to the current element being processed. It points to an element **εj** and is used in subsequent operations.\n- **e**: A stack used to store partial entity mentions. During entity recognition, the stack **e** holds the entities being recognized.\n- **β**: A buffer that stores unprocessed words.\n- **T**: Labeled trigger arcs. It stores the relationships between triggers and other elements.\n- **E**: Labeled entity mention arcs. It stores the relationships between entities.\n- **R**: A set of argument role arcs that represent the relationships between argument roles.\n- **A**: A stack used to store action history, tracking the actions that have been performed.\n\n### 2. Transition Operations:\nThese operations are performed during the state transition and affect the stack, queue, and buffer. Each operation is defined as follows:\n- **LEFT-PASS**:    - Adds an arc from the current element **λ(tj)** to the top element of the stack **σ(ei)**, used for generating semantic roles between triggers and entities.\n- **RIGHT-PASS**:    - Adds an arc from the top element of the stack **σ(ti)** to the current element **λ(ej)**, similarly used for generating semantic roles between triggers and entities.\n- **LEFT-RELATE**:    - Adds an arc from the current element **λ(ej)** to the top element of the stack **σ(ei)**, used for generating semantic relations between entities and entities.\n- **RIGHT-RELATE**:    - Adds an arc from the top element of the stack **σ(ei)** to the current element **λ(ej)**, similarly used for generating semantic relations between entities and entities.\n- **NO-PASS**:    - Performs this operation if no semantic role can be assigned between the current element **λ(εj)** and the top element of the stack **σ(εi)**, indicating no valid semantic role relationship.\n- **SHIFT**:    - Moves the current word from the buffer to the stack for processing. Used for handling words in the buffer.\n- **DUAL-SHIFT**:    - When the current element is both a trigger and the first word of an entity, this operation not only moves the current word to the stack but also copies it to the variable **λ** and pushes it to the buffer. This allows simultaneous processing of triggers and entities.\n- **DELETE**:    - Removes the top word from the buffer, indicating that the word no longer needs further processing.\n- **TRIGGER-GEN**:    - Moves the word from the buffer to the variable **λ** and assigns an event label, marking the word as an event trigger.\n- **ENTITY-SHIFT**:    - Moves the word from the buffer to the entity stack **e**, beginning the recognition of an entity.\n- **ENTITY-GEN**:    - Summarizes all elements in the entity stack **e** into a vector representation, assigns an entity label, and moves the representation to **λ**.\n- **ENTITY-BACK**:    - Pops all words off the entity stack **e** and pushes all except the bottom word back into the buffer.\n\n##Instructions:{}\nLabels include: {}.\n\n##Output format: \nBased on the definitions above, return the transition operations in a list. For example: \n ['ENTITY-SHIFT', 'ENTITY-GEN-per', 'ENTITY-BACK', 'SHIFT', 'O-DELETE', 'O-DELETE', 'TRIGGER-GEN-elect', 'NO-PASS', 'SHIFT', 'O-DELETE', 'O-DELETE', 'O-DELETE', 'ENTITY-SHIFT', 'ENTITY-GEN-per', 'ENTITY-BACK', 'RIGHT-PASS-person', 'NO-PASS', 'SHIFT', 'O-DELETE']\n\n"

    entity_instruction = []
    entity_format = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/train.jsonl'
    with open(entity_format, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datasets = json.loads(line)
            if datasets['instruction'] not in entity_instruction:
                entity_instruction.append(datasets['instruction'])
    print(len(list(set(entity_instruction))))

    event_instruction = []
    event_format = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/train.jsonl'
    with open(event_format, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data['instruction'] not in event_format:
                event_instruction.append(data['instruction'])
    print(len(list(set(event_instruction))))

    ner_paths_1 = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_res_2/train.txt', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_res_2/test.txt', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/transition_res_2/dev.txt',]

    for path in ner_paths_1:
        with open('/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/valid_pattern.json', 'r', encoding='utf-8') as f:
            valid_pattern = json.load(f)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            id = 0
            new_data = []
            for line in lines:
                line = json.loads(line)
                tmp = dict()
                tmp['instruction'] = instruction.format(entity_instruction[id % len(entity_instruction)], json.dumps(valid_pattern, ensure_ascii=False), )
                tmp['input'] = line['Sent']
                tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                new_data.append(tmp)
                id += 1
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
        with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                for data in new_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    ner_paths_2 =[
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_res_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_res_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/transition_res_2/dev.jsonl',]
    for path in ner_paths_2:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            id = 0
            new_data = []
            for line in lines:
                line = json.loads(line)
                tmp = dict()
                tmp['instruction'] = instruction.format(line['instruction'], json.dumps(line['schema'], ensure_ascii=False), )
                tmp['input'] = line['Sent']
                tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                new_data.append(tmp)
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
        with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                for data in new_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

    ner_paths_3=[
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_res_2/train.data', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_res_2/test.data', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/transition_res_2/dev.data']
    for path in ner_paths_3:
        with open('/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/valid_pattern.json', 'r', encoding='utf-8') as f:
            valid_pattern = json.load(f)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            id = 0
            new_data = []
            for line in lines:
                line = json.loads(line)
                tmp = dict()
                tmp['instruction'] = instruction.format(entity_instruction[id % len(entity_instruction)], json.dumps(valid_pattern, ensure_ascii=False), )
                tmp['input'] = line['Sent']
                tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                new_data.append(tmp)
                id += 1
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
        with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                for data in new_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
    
    event_paths_1 = [
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_res_2/train_english.oneie.json', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_res_2/test_english.oneie.json', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/transition_res_2/dev_english.oneie.json']

    event_paths_2 = [
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_res_2/train.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_res_2/test.jsonl', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/transition_res_2/dev.jsonl', \
    ]
    for path in event_paths_1:
        with open('/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/valid_pattern_.json', 'r', encoding='utf-8') as f:
            valid_pattern = json.load(f)
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            id = 0
            new_data = []
            for line in lines:
                line = json.loads(line)
                tmp = dict()
                valid = dict()
                valid['ent'] = valid_pattern['ner']
                valid['rel'] = valid_pattern['relation']
                valid['event'] = valid_pattern['event_role']
                tmp['instruction'] = instruction.format(event_instruction[id % len(event_instruction)] + ' Besides, ' + entity_instruction[id % len(entity_instruction)].lower(), json.dumps(valid, ensure_ascii=False), )
                tmp['input'] = line['Sent']
                tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                new_data.append(tmp)
                id += 1
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
        with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                for data in new_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

    for path in event_paths_2:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            id = 0
            new_data = []
            for line in lines:
                line = json.loads(line)
                tmp = dict()
                tmp['instruction'] = instruction.format(line['instruction'], json.dumps(line['schema'], ensure_ascii=False), )
                tmp['input'] = line['Sent']
                tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                new_data.append(tmp)
                id += 1
        if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
            os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
        with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                for data in new_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

    relation_path =[
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/transition_res_2', \
    '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/transition_res_2']
    for root in relation_path:
        for pa in ['train.jsonl', 'test.jsonl', 'dev.jsonl']:
            path = root + '/' + pa
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                id = 0
                new_data = []
                for line in lines:
                    line = json.loads(line)
                    tmp = dict()
                    tmp['instruction'] = instruction.format(line['instruction'], json.dumps(line['schema'], ensure_ascii=False), )
                    tmp['input'] = line['Sent']
                    tmp['output'] = json.dumps({'actions':line['actions']}, ensure_ascii=False)
                    new_data.append(tmp)
            if not os.path.exists('/'.join(path.split('/')[:-2]) + '/llama_v3/'):
                os.makedirs('/'.join(path.split('/')[:-2]) + '/llama_v3/')
            with open('/'.join(path.split('/')[:-2]) + '/llama_v3/' + path.split('/')[-1], 'w', encoding='utf-8') as f:
                    for data in new_data:
                        f.write(json.dumps(data, ensure_ascii=False) + '\n')
def merge2one():
    save_path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/v3.0_all'
    paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/llama_v3/train.txt',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/llama_v3/train_english.oneie.json', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/llama_v3/train.jsonl', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/llama_v3/train.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/llama_v3/train.data', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/llama_v3/train.jsonl']
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    all_dataset = []
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                all_dataset.append(line)
    print(len(all_dataset))
    with open(save_path + '/' + 'train.json', 'w', encoding='utf-8') as f:
        # for data in all_dataset:
        json.dump(all_dataset, f, ensure_ascii=False, indent=4)


# transfer2transition()
transfer2llama()
merge2one()
paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/llama_v3/test.txt',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/llama_v3/test_english.oneie.json', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/llama_v3/test.jsonl', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/llama_v3/test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/llama_v3/test.data', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/llama_v3/train.jsonl', \
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/llama_v3/train.jsonl']
save_path = [
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/ace2004_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/ace_add_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/conll03_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/casie_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/genia_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/14lap_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/14res_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/15res_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/16res_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/conll04_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/nyt_test_v3.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/scierc_test_v3.jsonl'
]
for id in range(len(paths)):
    datasets = []
    with open(paths[id], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            datasets.append(line)
    with open(save_path[id], 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)