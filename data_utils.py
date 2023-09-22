# @Time    : 2023/1/22 16:22
# @Author  : tk
# @FileName: data_utils.py
import copy
import json
import typing
import numpy as np
import torch
from deep_training.data_helper import DataHelper, ModelArguments, TrainingArguments, DataArguments
from fastdatasets.record import load_dataset as Loader, RECORD, WriterObject, gfile
from tqdm import tqdm
from transformers import HfArgumentParser,PreTrainedTokenizer
from aigc_zoo.model_zoo.moss.llm_model import MyTransformer,MossConfig,MossTokenizer,PetlArguments,PromptArguments
from data_processer import DataStrategy, TokenSupervision, TokenUnSupervision, TokenSupervisionRounds, \
    TokenRoundsForMoss

from config import *

data_conf = {
    'strategy': DataStrategy.mos_rounds,  # 数据策略选项
    DataStrategy.sup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.unsup: {
        'stride':  int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.sub_rounds: {
        'stride': int(train_info_args['max_seq_length'] / 3 * 2),
    },
    DataStrategy.mos_rounds: {
    }
}

class NN_DataHelper(DataHelper):
    index = 1
    def on_data_ready(self):
        self.index = -1

    # 切分词
    def on_data_process(self, data: typing.Any, mode: str):
        self.index += 1

        tokenizer: MossTokenizer
        config: MossConfig
        max_seq_length = self.max_seq_length_dict[mode]
        tokenizer = self.tokenizer # noqa
        config = self.config # noqa
        examples = data

        strategy = data_conf['strategy']
        if strategy == DataStrategy.sup:
            ds = TokenSupervision.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                          **data_conf[strategy])
        elif strategy == DataStrategy.unsup:
            ds = TokenUnSupervision.process(tokenizer, config=config, max_seq_length=max_seq_length, examples=examples,
                                            **data_conf[strategy])
        elif strategy == DataStrategy.sub_rounds:
            ds = TokenSupervisionRounds.process(tokenizer, config=config, max_seq_length=max_seq_length,
                                                examples=examples,
                                                **data_conf[strategy])
        elif strategy == DataStrategy.mos_rounds:
            ds = TokenRoundsForMoss.process(tokenizer, config=config, max_seq_length=max_seq_length,
                                                examples=examples,
                                                **data_conf[strategy])
        else:
            raise ValueError('Invalid strategy', strategy)
        if not ds:
            return None

        if self.index < 3:
            print(ds[0])
        return ds


    # 读取文件
    def on_get_corpus(self, files: typing.List, mode: str):
        D = []
        strategy = data_conf['strategy']
        for file in files:
            with open(file, mode='r', encoding='utf-8', newline='\n') as f:
                lines = f.readlines()

            for i, line in enumerate(lines):
                jd = json.loads(line)
                if not jd:
                    continue
                paragraph = jd['paragraph']
                if i < 10:
                    print(paragraph)
                sub = []
                # 自行做模板
                for session in paragraph:
                    a = session['a']
                    assert len(a), ValueError('answer cannot empty')
                    sub.append(session)
                if strategy == DataStrategy.mos_rounds:
                   D.append((jd['meta_instruction'],copy.deepcopy(sub)))
                else:
                    D.append(copy.deepcopy(sub))
                sub.clear()

        return D

    def collate_fn(self, batch):
        o = {}
        for i, b in enumerate(batch):
            if i == 0:
                for k in b:
                    o[k] = [torch.tensor(b[k])]
            else:
                for k in b:
                    o[k].append(torch.tensor(b[k]))
        for k in o:
            o[k] = torch.stack(o[k])

        maxlen = torch.max(o.pop('seqlen'))
        o['input_ids'] = o['input_ids'][:, :maxlen]
        o['attention_mask'] = o['attention_mask'][:, :maxlen]
        o['labels'] = o['labels'][:, :maxlen].long()
        return o

    def make_dataset_all(self):
        data_args = self.data_args

        # schema for arrow parquet
        schema = {
            "input_ids": "int32_list",
            "attention_mask": "int32_list",
            "labels": "int32_list",
            "seqlen": "int32_list",
        }
        # 缓存数据集
        if data_args.do_train:
            self.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train',
                                        schema=schema)
        if data_args.do_eval:
            self.make_dataset_with_args(data_args.eval_file, mode='eval', schema=schema)
        if data_args.do_test:
            self.make_dataset_with_args(data_args.test_file, mode='test', schema=schema)

if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, PetlArguments,PromptArguments))
    model_args, training_args, data_args, _,_ = parser.parse_dict(train_info_args)

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _,_ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer,config_class_name=MossConfig,config_kwargs={"torch_dtype": "float16"})
    

    # 检测是否存在 output/dataset_0-train.record ，不存在则制作数据集
    dataHelper.make_dataset_all()



    # def shuffle_records(record_filenames, outfile, compression_type='GZIP'):
    #     print('shuffle_records record...')
    #     options = RECORD.TFRecordOptions(compression_type=compression_type)
    #     dataset_reader = Loader.RandomDataset(record_filenames, options=options, with_share_memory=True)
    #     data_size = len(dataset_reader)
    #     all_example = []
    #     for i in tqdm(range(data_size), desc='load records'):
    #         serialized = dataset_reader[i]
    #         all_example.append(serialized)
    #     dataset_reader.close()
    #
    #     shuffle_idx = list(range(data_size))
    #     random.shuffle(shuffle_idx)
    #     writer = WriterObject(outfile, options=options)
    #     for i in tqdm(shuffle_idx, desc='shuffle record'):
    #         example = all_example[i]
    #         writer.write(example)
    #     writer.close()
    #
    #
    # # 对每个record 再次打乱
    # for filename in dataHelper.train_files:
    #     shuffle_records(filename, filename)
