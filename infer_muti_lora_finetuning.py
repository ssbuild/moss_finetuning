# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.moss import MossConfig
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer, MossTokenizer,LoraArguments,PromptArguments,LoraModel
from aigc_zoo.utils.moss_generate import Generate

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments,))
    (model_args, ) = parser.parse_dict(train_info_args, allow_extra_keys=True)

    

    dataHelper = NN_DataHelper(model_args)
    tokenizer: MossTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer, config_class_name=MossConfig,config_kwargs={"torch_dtype": "float16"})

    ckpt_dir = './best_ckpt/last'
    config = MossConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    lora_args = LoraArguments.from_pretrained(ckpt_dir)
    assert lora_args.inference_mode == True

    new_num_tokens = config.vocab_size
    if config.task_specific_params is not None and config.task_specific_params.get('vocab_size', None) is not None:
        config.vocab_size = config.task_specific_params['vocab_size']

    pl_model = MyTransformer(config=config, model_args=model_args, lora_args=lora_args,
                             torch_dtype=torch.float16,new_num_tokens=new_num_tokens,
                             # load_in_8bit=global_args["load_in_8bit"],
                             # # device_map="auto",
                             # device_map = {"":0} # 第一块卡
                             )
    # 加载多个lora权重
    pl_model.load_sft_weight(ckpt_dir, adapter_name="default")

    # 加载多个lora权重
    # pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")

    # 加载多个lora权重
    # pl_model.load_sft_weight(ckpt_dir, adapter_name="yourname")

    pl_model.eval().half().cuda()



    # backbone model replaced LoraModel
    lora_model: LoraModel = pl_model.backbone

    gen_core = Generate(lora_model, tokenizer)
    query = "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:"

    # 基准模型推理
    with lora_model.disable_adapter():

        response = gen_core.chat(query, max_length=512,
                                 # do_sample=False, top_p=0.7, temperature=0.95,
                                 )
        print(query, ' 返回: ', response)

    lora_model.set_adapter(adapter_name='default')

    response = gen_core.chat(query, max_length=512,
                             # do_sample=False, top_p=0.7, temperature=0.95,
                             )

    print(query, ' 返回: ', response)