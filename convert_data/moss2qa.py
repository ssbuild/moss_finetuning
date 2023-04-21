# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 15:44
import json
import os

root_dir = r'E:\ai_engine\MOSS\SFT_data\conversations\conversation_without_plugins'
output_file = '../data/train.json'

filenames = []
for root,path_lists,files in os.walk(root_dir):
    for path in files:
        if path.endswith('.json'):
            filenames.append(os.path.join(root,path))

D = []
for file in filenames:
    with open(file,mode='r',encoding='utf-8') as f:
        jd = json.loads(f.read())

        num_turns = jd['num_turns']
        chats = jd['chat']

        paragraph = []
        for i in range(num_turns):
            chat = chats['turn_{}'.format(i+1)]
            chat['q'] = chat.pop('Human')
            chat['a'] = chat.pop('MOSS')
            paragraph.append(chat)


        D.append({
            'paragraph': paragraph,
            'meta_instruction': jd['meta_instruction']
        })

with open(output_file,mode='w',encoding='utf-8',newline='\n') as f:
    for d in D:
        f.write(json.dumps(d,ensure_ascii=False) + '\n')