# -*- coding: utf-8 -*-
# @Time    : 2023/5/16 10:07

import json
import os
import torch
from transformers import BitsAndBytesConfig
from config.constant_map import train_info_models, train_target_modules_maps

# 量化权重不支持此模式训练
train_model_config = train_info_models['moss-moon-003-sft']

global_args = {
    "load_in_8bit": False,
    "load_in_4bit": False,

    # load_in_4bit 量化配置
    "quantization_config": BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ),
    "config_merge": {

    },
    "n_layer": -1,  # 是否使用骨干网络的全部层数 最大1-34， -1 表示全层, 否则只用只用N层
    "num_layers_freeze": -1,  # 非lora,非p-tuning 模式 ， <= config.json num_layers
}





lora_info_args = {
    'with_lora': True,  # 是否启用lora模块
    'r': 8,
    'target_modules': train_target_modules_maps[train_model_config['model_type']],
    'target_dtype': 16, # 半精度
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
}

adalora_info_args = {
    'with_lora': False,  # 是否启用adalora模块
    'r': 8,
    'target_modules': train_target_modules_maps[train_model_config['model_type']],
    'target_dtype': 16, # 半精度
    'lora_alpha': 32,
    'lora_dropout': 0.1,
    'bias': 'none',  # Bias type for Lora. Can be 'none', 'all' or 'lora_only'"
    'modules_to_save' : None, # "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "

    'target_r':8, # Target Lora matrix dimension.
    'init_r': 12, #Intial Lora matrix dimension.
    'tinit': 0, #The steps of initial warmup.
    'tfinal': 0, #The steps of final warmup.
    'deltaT': 1, #Step interval of rank allocation.
    'beta1': 0.85, #Hyperparameter of EMA.
    'beta2': 0.85, #Hyperparameter of EMA.
    'orth_reg_weight': 0.5, #The orthogonal regularization coefficient.
    'total_step': None, #The total training steps.
    'rank_pattern': None, #The saved rank pattern.
}


train_info_args = {
    'devices': 1,
    'data_backend': 'record',  #one of record lmdb, 超大数据集可以使用 lmdb , 注 lmdb 存储空间比record大
    # 预训练模型路径
    **train_model_config,

    'convert_onnx': False, # 转换onnx模型
    'do_train': True,
    'train_file':  [ './data/finetune_train_examples.json'],
    'max_epochs': 20,
    'max_steps': -1,

    # lamb,adma,adamw_hf,adam,adamw,adamw_torch,adamw_torch_fused,adamw_torch_xla,adamw_apex_fused,
    # adafactor,adamw_anyprecision,sgd,adagrad,adamw_bnb_8bit,adamw_8bit,lion,lion_8bit,lion_32bit,
    # paged_adamw_32bit,paged_adamw_8bit,paged_lion_32bit,paged_lion_8bit,
    # lamb_fused_dp adagrad_cpu_dp adam_cpu_dp adam_fused_dp
    'optimizer': 'lion',


    'scheduler_type': 'CAWR', #one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau, cosine,cosine_with_restarts,polynomial,constant,constant_with_warmup,inverse_sqrt,reduce_lr_on_plateau]
    'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 0.5, 'verbose': False},

    # 'scheduler_type': 'linear',# one of [linear,WarmupCosine,CAWR,CAL,Step,ReduceLROnPlateau
    # 'scheduler': None,

    # 切换scheduler类型
    # 'scheduler_type': 'WarmupCosine',
    # 'scheduler': None,

    # 'scheduler_type': 'ReduceLROnPlateau',
    # 'scheduler': None,

    # 'scheduler_type': 'Step',
    # 'scheduler':{ 'decay_rate': 0.999,'decay_steps': 100,'verbose': True},

    # 'scheduler_type': 'CAWR',
    # 'scheduler':{'T_mult': 1, 'rewarm_epoch_num': 2, 'verbose': True},

    # 'scheduler_type': 'CAL',
    # 'scheduler': {'rewarm_epoch_num': 2,'verbose': True},


    'optimizer_betas': (0.9, 0.999),
    'train_batch_size': 2,
    'eval_batch_size': 2,
    'test_batch_size': 2,
    'learning_rate': 2e-4,  #
    'adam_epsilon': 1e-8,
    'gradient_accumulation_steps': 1,
    'max_grad_norm': 1.0,
    'weight_decay': 0,
    'warmup_steps': 0,
    'output_dir': './output',
    'max_seq_length': 1024, # 如果资源充足，推荐长度2048
    'max_target_length': 100,  # 预测最大长度, 保留字段
    'use_fast_tokenizer': False,
    'do_lower_case': False,

    ##############  lora模块
    #注意lora,adalora 和 ptuning-v2 禁止同时使用
   'lora': {**lora_info_args},
   'adalora': {**adalora_info_args},

}

