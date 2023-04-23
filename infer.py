# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
from deep_training.data_helper import ModelArguments, TrainingArguments, DataArguments
from deep_training.nlp.models.lora.v2 import LoraArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer,MossConfig,MossTokenizer

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments))
    model_args, training_args, data_args, _ = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer: MossConfig
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(
        tokenizer_class_name=MossTokenizer, config_class_name=MossConfig)
    config.torch_dtype = "float16"
    # config.n_layer = 1

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
    model = pl_model.get_llm_model()
    model.half().cuda()
    model = model.eval()

    query =  "<|Human|>: 你好<eoh>\n<|MOSS|>:"
    response = model.chat(tokenizer, query, max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print(query,' 返回: ',response)

    query = response + "\n<|Human|>: 推荐五部科幻电影<eoh>\n<|MOSS|>:"
    response = model.chat(tokenizer, query, max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print(query, ' 返回: ', response)

