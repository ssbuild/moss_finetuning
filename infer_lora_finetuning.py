# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os

import torch
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.moss import MossConfig
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper,global_args
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer, MossTokenizer,LoraArguments,PromptArguments
from aigc_zoo.utils.moss_generate import Generate

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments,))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)

    

    dataHelper = NN_DataHelper(model_args, None, data_args)
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
    # 加载lora权重
    pl_model.load_sft_weight(ckpt_dir)

    pl_model.eval().half().cuda()

    enable_merge_weight = False
    if enable_merge_weight:
        # 合并lora 权重 保存
        pl_model.save_pretrained_merge_lora(os.path.join(ckpt_dir, 'pytorch_model_merge.bin'))
    else:
        model = pl_model.get_llm_model()

        gen_core = Generate(model, tokenizer)

        query = "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:"
        response = gen_core.chat(query, max_length=2048,
                              # do_sample=False, top_p=0.7, temperature=0.95,
                              )
        print(query, ' 返回: ', response)

        # query = response + "\n<|Human|>: 推荐五部科幻电影<eoh>\n<|MOSS|>:"
        # response = model.chat(tokenizer, query, max_length=2048,
        #                       eos_token_id=config.eos_token_id,
        #                       do_sample=True, top_p=0.7, temperature=0.95,
        #                       )
        # print(query, ' 返回: ', response)
