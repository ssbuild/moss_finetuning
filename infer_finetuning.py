# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os
import re
from collections import OrderedDict

import torch
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper, get_deepspeed_config
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer,MossTokenizer,MossConfig
from aigc_zoo.utils.moss_generate import Generate

deep_config = get_deepspeed_config()


if __name__ == '__main__':
    train_info_args['seed'] = None
    train_info_args['model_name_or_path'] = None

    parser = HfArgumentParser((ModelArguments, DataArguments,))
    model_args, data_args = parser.parse_dict(train_info_args, allow_extra_keys=True)
    

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: MossTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer, config_class_name=MossConfig,config_kwargs={"torch_dtype": "float16"})
    ###################### 注意 选最新权重
    #选择最新的权重 ， 根据时间排序 选最新的
    config = MossConfig.from_pretrained('./best_ckpt')
    config.initializer_weight = False

    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,)
    if deep_config is None:
        train_weight = './best_ckpt/last-v3.ckpt'
        assert os.path.exists(train_weight)
    else:
        #使用转换脚本命令 生成 ./best_ckpt/last/best.pt 权重文件
        # cd best_ckpt/last
        # python ./zero_to_fp32.py . best.pt
        train_weight = './best_ckpt/last/best.pt'

    pl_model.load_sft_weight(train_weight,strict=False)

    model = pl_model.get_llm_model()
    model.eval().half().cuda()

    gen_core = Generate(model, tokenizer)

    query = "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:"
    response = gen_core.chat(query,  max_length=2048,
                          # do_sample=False, top_p=0.7, temperature=0.95,
                          )
    print(query, ' 返回: ', response)

    # query = response + "\n<|Human|>: 推荐五部科幻电影<eoh>\n<|MOSS|>:"
    # response = model.chat(tokenizer, query, max_length=2048,
    #                       eos_token_id=config.eos_token_id,
    #                       do_sample=True, top_p=0.7, temperature=0.95,
    #                       )
    # print(query, ' 返回: ', response)

