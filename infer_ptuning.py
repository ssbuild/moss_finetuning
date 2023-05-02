# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
import os

from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.moss import MossConfig
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer, MossTokenizer,LoraArguments,PromptArguments

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PromptArguments))
    model_args, training_args, data_args,_, _ = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: MossTokenizer
    tokenizer, _, _, _ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=MossTokenizer, config_class_name=MossConfig)

    ckpt_dir = './best_ckpt'
    config = MossConfig.from_pretrained(ckpt_dir)
    config.initializer_weight = False
    prompt_args = PromptArguments.from_pretrained(ckpt_dir)
    assert prompt_args.inference_mode == True

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,prompt_args=prompt_args)
    # 加载lora权重
    pl_model.backbone.from_pretrained(pl_model.backbone.model, pretrained_model_name_or_path=ckpt_dir, prompt_config=prompt_args)
    pl_model.eval().half().cuda()

    model = pl_model.get_llm_model()
    query = "<|Human|>: 你好<eoh>\n<|MOSS|>:"
    response = model.chat(tokenizer, query,  max_length=2048,
                          eos_token_id=config.eos_token_id,
                          do_sample=True, top_p=0.7, temperature=0.95,
                          )
    print(query, ' 返回: ', response)

    query = response + "\n<|Human|>: 推荐五部科幻电影<eoh>\n<|MOSS|>:"
    response = model.chat(tokenizer, query, max_length=2048,
                          eos_token_id=config.eos_token_id,
                          do_sample=True, top_p=0.7, temperature=0.95,
                          )
    print(query, ' 返回: ', response)
