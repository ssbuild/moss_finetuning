# -*- coding: utf-8 -*-
import os
import torch
from deep_training.data_helper import ModelArguments, DataArguments, TrainingArguments
from deep_training.trainer.pl.modelcheckpoint import ModelCheckpointEx
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.strategies import DeepSpeedStrategy
from transformers import HfArgumentParser

from config import global_args
from data_utils import NN_DataHelper, train_info_args, get_deepspeed_config
from models import MyTransformer,MossTokenizer,MossConfig,LoraArguments,PromptArguments


if __name__ == '__main__':
    parser = HfArgumentParser((ModelArguments, TrainingArguments, DataArguments, LoraArguments,PromptArguments))
    model_args, training_args, data_args, lora_args,prompt_args = parser.parse_dict(train_info_args)
    lora_args = lora_args.config
    prompt_args = prompt_args.config

    output_weight_dir = './best_ckpt'

    config_kwargs = {"torch_dtype": "float16"}
    if global_args["n_layer"] > 0:
        config_kwargs["n_layer"] = global_args["n_layer"]

    dataHelper = NN_DataHelper(model_args, training_args, data_args)
    tokenizer, config, _, _ = dataHelper.load_tokenizer_and_config(tokenizer_class_name=MossTokenizer,
                                                                   config_class_name=MossConfig,
                                                                   config_kwargs=config_kwargs)

    # 缓存数据集
    if data_args.do_train:
        dataHelper.make_dataset_with_args(data_args.train_file, mixed_data=False, shuffle=True, mode='train')
    if data_args.do_eval:
        dataHelper.make_dataset_with_args(data_args.eval_file, mode='eval')
    if data_args.do_test:
        dataHelper.make_dataset_with_args(data_args.test_file, mode='test')

    checkpoint_callback = ModelCheckpointEx(
        # monitor='loss',
        dirpath=output_weight_dir,
        save_weights_only=True,
        save_last=True,
        save_top_k=1,
        # every_n_train_steps=2000 // training_args.gradient_accumulation_steps,
        every_n_epochs=1,
        lora_args=lora_args,
        prompt_args=prompt_args,
    )

    deepspeed_config = get_deepspeed_config()
    strategy = 'ddp' if torch.cuda.device_count() > 1 else 'auto'
    if deepspeed_config is not None and len(deepspeed_config):
        strategy = DeepSpeedStrategy(config=deepspeed_config, )
    trainer = Trainer(
        callbacks=[checkpoint_callback,LearningRateMonitor(logging_interval='step')],
        max_epochs=training_args.max_epochs,
        max_steps=training_args.max_steps,
        accelerator="gpu",
        devices=data_args.devices,
        enable_progress_bar=True,
        default_root_dir=data_args.output_dir,
        gradient_clip_val=training_args.max_grad_norm,
        accumulate_grad_batches=training_args.gradient_accumulation_steps,
        num_sanity_val_steps=0,
        strategy=strategy,
        precision='16' ,# 可以自行尝试  "32": "32-true", "16": "16-mixed", "bf16": "bf16-mixed"
    )


    pl_model = MyTransformer(config=config, model_args=model_args, training_args=training_args,lora_args=lora_args,prompt_args=prompt_args,
                             quantization_config=global_args["quantization_config"],
                             num_layers_freeze=global_args["num_layers_freeze"],
                             load_in_8bit=global_args["load_in_8bit"],
                             device_map={"": trainer.local_rank} if trainer.world_size > 1 else "auto",
                             torch_dtype=torch.float16,
                             new_num_tokens=len(tokenizer),  # 可能扩充词
                             )

    config.save_pretrained(output_weight_dir)

    # 加载sft权重
    # pl_model.load_sft_weight('./best_ckpt/best.pt',is_trainable=True)

    pl_model.float()


    def dataset_loader_filter_fn(dataset):
        print('*' * 30, 'total count', len(dataset))
        return dataset


    train_datasets = dataHelper.load_distributed_random_sampler(
        dataHelper.train_files,
        with_load_memory=data_args.data_backend == 'record',
        collate_fn=dataHelper.collate_fn,
        batch_size=training_args.train_batch_size,
        drop_last=True,  # 多卡建议扔掉
        num_processes=trainer.world_size, process_index=trainer.global_rank,
        dataset_loader_filter_fn=dataset_loader_filter_fn,
        num_workers=0
    )

    trainer.fit(pl_model, train_dataloaders=train_datasets)
