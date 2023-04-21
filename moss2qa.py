# -*- coding: utf-8 -*-
# @Time    : 2023/4/21 15:44
import json
import os

root_dir = r'E:\ai_engine\MOSS\SFT_data\conversations\conversation_without_plugins'

filenames = []
for root,path_lists,files in os.walk(root_dir):
    for path in files:
        if path.endswith('.json'):
            filenames.append(os.path.join(root,path))

for file in filenames:
    with open(file,mode='r',encoding='utf-8') as f:
        jd = json.loads(f.read())

        print(jd)