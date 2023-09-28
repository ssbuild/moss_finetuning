# -*- coding: utf-8 -*-
# @Time:  23:20
# @Author: tk
# @File：model_maps
from aigc_zoo.constants.define import (TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING,
                                       TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING)

__all__ = [
    "TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING",
    "TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING",
    "train_model_config"
]

train_info_models = {
    'moss-moon-003-sft': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft',
    },
    'moss-moon-003-sft-plugin': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin',
    },
    # 官方量化
    'moss-moon-003-sft-int4-gptq': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4-gptq',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4-gptq/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4-gptq',
    },

    # 官方量化
    'moss-moon-003-sft-plugin-int4-gptq': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4-gptq',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4-gptq/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4-gptq',
    },

    'moss-moon-003-sft-int4': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-int4',
    },

    'moss-moon-003-sft-plugin-int4': {
        'model_type': 'moss',
        'model_name_or_path': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4',
        'config_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4/config.json',
        'tokenizer_name': '/data/nlp/pre_models/torch/moss/moss-moon-003-sft-plugin-int4',
    },

}

# 按需修改
# TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_ADALORA_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
# TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

train_model_config = train_info_models['moss-moon-003-sft']