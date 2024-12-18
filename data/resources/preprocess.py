import json
import itertools
import os

'''
{"id": "entity.conll03.test.0", "instruction": "Can you detect any entities that may be present in the given text and classify them based on their types?", "schema": {"ent": ["miscellaneous", "person", "location", "organization"], "rel": [], "event": {}}, "ans": {"ent": [{"type": "person", "text": "CHINA", "span": [30, 35]}, {"type": "location", "text": "JAPAN", "span": [8, 13]}], "rel": [], "event": []}, "text": "SOCCER - JAPAN GET LUCKY WIN , CHINA IN SURPRISE DEFEAT .", "bg": ""}

{"id": "event.casie.test.0", "instruction": "Retrieve pertinent event details from the provided text, encompassing the trigger, type, and arguments.", "schema": {"ent": ["geopolitical entity", "time", "file", "website", "data", "common vulnerabilities and exposures", "money", "patch", "malware", "person", "purpose", "number", "vulnerability", "version", "capabilities", "payment method", "system", "software", "personally identifiable information", "organization", "device"], "rel": [], "event": {"phishing": ["purpose", "time", "victim", "trusted entity", "place", "attack pattern", "attacker", "damage amount", "tool"], "databreach": ["time", "purpose", "compromised data", "number of victim", "victim", "number of data", "place", "attack pattern", "attacker", "damage amount", "tool"], "ransom": ["time", "victim", "price", "place", "attack pattern", "attacker", "payment method", "damage amount", "tool"], "discover vulnerability": ["time", "discoverer", "vulnerable system version", "supported platform", "common vulnerabilities and exposures", "vulnerability", "capabilities", "vulnerable system", "vulnerable system owner"], "patch vulnerability": ["time", "releaser", "patch number", "vulnerable system version", "common vulnerabilities and exposures", "supported platform", "issues addressed", "patch", "vulnerability", "vulnerable system"]}}, "ans": {"ent": [{"type": "organization", "text": "UConn Health", "span": [122, 134]}, {"type": "number", "text": "326,000", "span": [152, 159]}, {"type": "person", "text": "individuals", "span": [160, 171]}], "rel": [], "event": [{"event_type": "phishing", "trigger": {"text": "Phishing", "span": [0, 8]}, "args": []}, {"event_type": "databreach", "trigger": {"text": "health data breaches", "span": [80, 100]}, "args": [{"role": "victim", "text": "UConn Health", "span": [122, 134]}, {"role": "number of victim", "text": "326,000", "span": [152, 159]}, {"role": "victim", "text": "individuals", "span": [160, 171]}]}]}, "text": "Phishing and other hacking incidents have led to several recently reported large health data breaches , including one that UConn Health reports affected 326,000 individuals .", "bg": ""}

{"id": "relation.conll04.test.0", "instruction": "How do two words relate to each other based on the sentence that describes them?", "schema": {"ent": ["people", "location", "organization", "other"], "rel": ["located in", "organization in", "live in", "work for", "kill"], "event": {}}, "ans": {"ent": [{"type": "organization", "text": "Hakawati Theatre", "span": [21, 37]}, {"type": "other", "text": "Arab", "span": [41, 45]}, {"type": "location", "text": "Jerusalem", "span": [51, 60]}, {"type": "other", "text": "Palestinians", "span": [90, 102]}], "rel": [{"relation": "organization in", "head": {"text": "Hakawati Theatre", "span": [21, 37]}, "tail": {"text": "Jerusalem", "span": [51, 60]}}], "event": []}, "text": "An art exhibit at the Hakawati Theatre in Arab east Jerusalem was a series of portraits of Palestinians killed in the rebellion .", "bg": ""}

'''
def run():
    entity_instruction = []
    entity_format = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/train.jsonl'
    with open(entity_format, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            datasets = json.loads(line)
            if datasets['instruction'] not in entity_instruction:
                entity_instruction.append(datasets['instruction'])
    print(len(list(set(entity_instruction))))

    relation_instruction = []
    relation_format = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/train.jsonl'
    with open(relation_format, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data['instruction'] not in relation_format:
                relation_instruction.append(data['instruction'])
    print(len(list(set(relation_instruction))))

    event_instruction = []
    event_format = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/train.jsonl'
    with open(event_format, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            data = json.loads(line)
            if data['instruction'] not in event_format:
                event_instruction.append(data['instruction'])
    print(len(list(set(event_instruction))))

    ace04_path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/'
    acec05_path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/'
    genia_path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/'

    with open(ace04_path+'valid_pattern.json', 'r', encoding='utf-8') as f:
        valid_pattern = json.load(f)
    for file in ['train.txt', 'test.txt', 'dev.txt']:
        with open(ace04_path+file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            instruction_data = []
            id = 0
            for index in range(0, len(lines), 4):
                sentence = lines[index].strip()
                tags = lines[index + 2].strip().split('|')
                tmp = dict()
                tmp['id'] = "entity." + "ace04." + file.split('.')[0] + '.' + str(id)
                tmp['instruction'] = entity_instruction[id % len(entity_instruction)]
                tmp['schema'] = dict()
                tmp['schema']['ent'] = valid_pattern['ner']
                tmp['schema']['rel'] = valid_pattern['relation']
                tmp['schema']['event'] = dict()
                tmp['ans'] = dict()
                tmp['ans']['ent'] = []
                for tag in tags:
                    if tag:
                        start, end = tag.split(' ')[0].split(',')
                        label = tag.split(' ')[1]
                        tmp['ans']['ent'].append({'type':label, 'text':' '.join((sentence.split()[int(start):int(end)])), 'span':[int(start), int(end)]})
                tmp['ans']['rel'] = []
                tmp['ans']['event'] = []
                tmp['text'] = sentence
                tmp['bg'] = ""
                instruction_data.append(tmp)
                id += 1
        with open(ace04_path + file.split('.')[0] + '.jsonl', 'w', encoding='utf-8') as f:
            for data in instruction_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    with open(genia_path+'valid_pattern.json', 'r', encoding='utf-8') as f:
        valid_pattern = json.load(f)
    for file in ['train.data', 'test.data', 'dev.data']:
        with open(genia_path+file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            instruction_data = []
            id = 0
            for index in range(0, len(lines), 4):
                sentence = lines[index].strip()
                tags = lines[index + 2].strip().split('|')
                tmp = dict()
                tmp['id'] = "entity." + "genia." + file.split('.')[0] + '.' + str(id)
                tmp['instruction'] = entity_instruction[id % len(entity_instruction)]
                tmp['schema'] = dict()
                tmp['schema']['ent'] = valid_pattern['ner']
                tmp['schema']['rel'] = valid_pattern['relation']
                tmp['schema']['event'] = dict()
                tmp['ans'] = dict()
                tmp['ans']['ent'] = []
                for tag in tags:
                    if tag:
                        start, end = tag.split(' ')[0].split(',')
                        label = tag.split(' ')[1]
                        tmp['ans']['ent'].append({'type':label, 'text':' '.join((sentence.split()[int(start):int(end)])), 'span':[int(start), int(end)]})
                tmp['ans']['rel'] = []
                tmp['ans']['event'] = []
                tmp['text'] = sentence
                tmp['bg'] = ""
                instruction_data.append(tmp)
                id += 1
        with open(genia_path + file.split('.')[0] + '.jsonl', 'w', encoding='utf-8') as f:
            for data in instruction_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

    # event_relation_format = [[(event, relation) for event in event_instruction] for relation in  relation_instruction]
    # print(event_relation_format)
    with open(acec05_path+'valid_pattern_.json', 'r', encoding='utf-8') as f:
        valid_pattern = json.load(f)
    for file in ['train_english.oneie.json', 'test_english.oneie.json', 'dev_english.oneie.json']:
        with open(acec05_path+file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            instruction_data = []
            id = 0
            for line in lines:
                line = json.loads(line)
                sentence = line['sentence']
                tmp = dict()
                tmp['id'] = "event.realtion." + "ace05." + file.split('.')[0] + '.' + str(id)
                tmp['instruction'] = ' '.join([event_instruction[id % len(event_instruction)], relation_instruction[id % len(relation_instruction)]])
                tmp['schema'] = dict()
                tmp['schema']['ent'] = valid_pattern['ner']
                tmp['schema']['rel'] = valid_pattern['relation']
                tmp['schema']['event'] = valid_pattern['event_role']
                tmp['ans'] = dict()
                tmp['ans']['ent'] = []
                tmp['ans']['rel'] = []
                tmp['ans']['event'] = []
                ner_dict = dict()
                for tag in line['entity_mentions']:
                    tmp['ans']['ent'].append({'type':tag['entity_type'], 'text':tag['text'], 'span':[int(tag["start"]), int(tag['end'])]})
                    ner_dict[tag['id']] = [int(tag["start"]), int(tag['end'])]
                for  tag in line['relation_mentions']:
                    if ner_dict[tag['arguments'][0]['entity_id']][0] <= ner_dict[tag['arguments'][1]['entity_id']][0]:
                        tmp['ans']['rel'].append({'relation':tag['relation_type'], 'head':{'text':tag['arguments'][0]['text'], 'span':ner_dict[tag['arguments'][0]['entity_id']]}, 'tail':{'text':tag['arguments'][1]['text'], 'span':ner_dict[tag['arguments'][1]['entity_id']]}})
                    else:
                        tmp['ans']['rel'].append({'relation':tag['relation_type'], 'head':{'text':tag['arguments'][1]['text'], 'span':ner_dict[tag['arguments'][1]['entity_id']]}, 'tail':{'text':tag['arguments'][0]['text'], 'span':ner_dict[tag['arguments'][0]['entity_id']]}})
                for tag in line['event_mentions']:
                    tmp['ans']['event'].append({'event_type':tag['event_type'], 'trigger':{'text':tag['trigger']['text'], 'span':[int(tag['trigger']['start']), int(tag['trigger']['end'])]}, 'args':[
                        {'role':args['role'], 'text':args['text'], 'span':ner_dict[args['entity_id']]} for args in tag['arguments']
                    ]})
                tmp['text'] = sentence
                tmp['bg'] = ""
                instruction_data.append(tmp)
                id += 1
        with open(acec05_path + file.split('.')[0] + '1109.jsonl', 'w', encoding='utf-8') as f:
            for data in instruction_data:
                f.write(json.dumps(data, ensure_ascii=False) + '\n')

def transfer2llama():
    paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot', 
            # '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc']

    for path in paths:
        for file in ['train.jsonl', 'test.jsonl', 'dev.jsonl']:
            all_dataset = []
            with open(path + '/' + file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    all_dataset.append({'instruction':line['instruction'] + '\n' + json.dumps(line['schema'], ensure_ascii=False), 'input':line['text'],'output':json.dumps(line['ans'], ensure_ascii=False)})
            if not os.path.exists(path + '/' + 'llama/'):
                os.mkdir(path + '/' + 'llama/')
            with open(path + '/' + 'llama/' + file, 'w', encoding='utf-8') as f:
                for data in all_dataset:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
    path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add'
    for file in ['train_english1109.jsonl', 'test_english1109.jsonl', 'dev_english1109.jsonl']:
        all_dataset = []
        with open(path + '/' + file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                line = json.loads(line)
                all_dataset.append({'instruction':line['instruction'] + '\n' + json.dumps(line['schema'], ensure_ascii=False), 'input':line['text'],'output':json.dumps(line['ans'], ensure_ascii=False)})
        if not os.path.exists(path + '/' + 'llama/'):
            os.mkdir(path + '/' + 'llama/')
        if 'train' in file:
            with open(path + '/' + 'llama/train.jsonl', 'w', encoding='utf-8') as f:
                for data in all_dataset:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif 'test' in file:
            with open(path + '/' + 'llama/test.jsonl', 'w', encoding='utf-8') as f:
                for data in all_dataset:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
        elif 'dev' in file:
            with open(path + '/' + 'llama/dev.jsonl', 'w', encoding='utf-8') as f:
                for data in all_dataset:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

def merge2one():
    save_path = '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/v1.0_all'
    paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie', 
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc']
    for file in ['train.jsonl', 'test.jsonl', 'dev.jsonl']:
        all_dataset = []
        for path in paths:
            with open(path + '/llama/' + file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = json.loads(line)
                    all_dataset.append(line)
        print(len(all_dataset))
        with open(save_path + '/' + file, 'w', encoding='utf-8') as f:
            # for data in all_dataset:
            json.dump(all_dataset, f, ensure_ascii=False, indent=4)

# run()
# transfer2llama()
# merge2one()

paths = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14lap/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/14res/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/15res/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/absa/16res/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ace2004/pot/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/EE/ACE_add/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/ent/conll03/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/event/casie/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/genia/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/conll04/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/nyt/llama/test.jsonl',
        '/data/liuweichang/workspace/LLaMA-Factory-main/data/resources/uie/rel/scierc/llama/test.jsonl']
save_path = ['/data/liuweichang/workspace/LLaMA-Factory-main/data/14lap_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/14res_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/15res_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/16res_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/ace2004_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/ace_add_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/conll03_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/casie_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/genia_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/conll04_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/nyt_test.jsonl',
            '/data/liuweichang/workspace/LLaMA-Factory-main/data/scierc_test.jsonl']
for id in range(len(paths)):
    datasets = []
    with open(paths[id], 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line)
            datasets.append(line)
    with open(save_path[id], 'w', encoding='utf-8') as f:
        json.dump(datasets, f, ensure_ascii=False, indent=4)
