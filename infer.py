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

    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args)
    model = pl_model.get_llm_model()
    model.half().cuda()
    model = model.eval()

    # 注意 长度不等于2048 会影响效果
    response, history = model.chat(tokenizer, "你好", history=[],max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print('你好',' ',response)

    response, history = model.chat(tokenizer, "晚上睡不着应该怎么办", history=history,max_length=2048,
                                   eos_token_id=config.eos_token_id,
                                   do_sample=True, top_p=0.7, temperature=0.95,
                                   )
    print('晚上睡不着应该怎么办',' ',response)

    # response, history = base_model.chat(tokenizer, "写一个诗歌，关于冬天", history=[],max_length=30)
    # print('写一个诗歌，关于冬天',' ',response)

