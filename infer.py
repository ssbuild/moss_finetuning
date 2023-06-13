# -*- coding: utf-8 -*-
# @Time    : 2023/3/9 15:29
from deep_training.data_helper import ModelArguments, DataArguments
from transformers import HfArgumentParser

from data_utils import train_info_args, NN_DataHelper
from models import MyTransformer,MossConfig,MossTokenizer

if __name__ == '__main__':
    train_info_args['seed'] = None
    parser = HfArgumentParser((ModelArguments, DataArguments, ))
    model_args, data_args = parser.parse_dict(train_info_args,allow_extra_keys=True)

    dataHelper = NN_DataHelper(model_args, None, data_args)
    tokenizer: MossTokenizer
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer, config_class_name=MossConfig,config_kwargs={"torch_dtype": "float16"})

    pl_model = MyTransformer(config=config, model_args=model_args, torch_dtype=torch.float16,)
    model = pl_model.get_llm_model()
    model.eval().half().cuda()

    query =  "<|Human|>: 如果一个女性想要发展信息技术行业，她应该做些什么？<eoh>\n<|MOSS|>:"
    response = model.chat(tokenizer, query, max_length=2048,
                          # do_sample=False, top_p=0.7, temperature=0.95,
                          )
    print(query,' 返回: ',response)

    # query = response + "\n<|Human|>: 推荐五部科幻电影<eoh>\n<|MOSS|>:"
    # response = model.chat(tokenizer, query, max_length=2048,
    #
    #
    #                                eos_token_id=config.eos_token_id,
    #                                do_sample=True, top_p=0.7, temperature=0.95,
    #                                )
    # print(query, ' 返回: ', response)

