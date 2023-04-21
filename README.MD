

## 1.安装
  - pip install -i https://pypi.org/simple -U deep_training>=0.1.3rc0 deepspeed
    

## 2.更新详情
- [deep_training](https://github.com/ssbuild/deep_training)

## 3.深度学习常规任务例子

- [pytorch-task-example](https://github.com/ssbuild/pytorch-task-example)
- [tf-task-example](https://github.com/ssbuild/tf-task-example)
- [llm_finetuning](https://github.com/ssbuild/llm_finetuning)
- [llm_rlhf_finetuning](https://github.com/ssbuild/llm_rlhf_finetuning)
- [chatglm_rlhf_finetuning](https://github.com/ssbuild/chatglm_rlhf_finetuning)



## 4.预训练权重

- [moss-moon-003-sft](https://huggingface.co/fnlp/moss-moon-003-sft)
- [moss-moon-003-sft-plugin](https://huggingface.co/fnlp/moss-moon-003-sft-plugin)
    




## 5.数据示例
    给出一种qa 数据示例.
    单条数据示例1
    {
        "id": 0, "paragraph": [
            #一轮会话
            {
                "q": "从南京到上海的路线",
                "a": [
                    "你好，南京到上海的路线如下：",
                    "1. 南京到上海，可以乘坐南京地铁1号线，在南京站乘坐轨道交通1号线。",
                    "2. 南京到浦东机场，可以搭乘上海地铁1号，在陆家嘴站乘坐地铁1线，在浦东国际机场站乘坐机场快线，前往上海浦东国际机场。",
                    "3. 上海到南京，可以换乘上海地铁2号线，从南京站换乘地铁2线，再从南京南站换乘地铁1路，然后到达上海站"
                ]
            }
            #二轮....
        ]
    }
    单条数据示例2
    {"id": 0, "paragraph": [

        {
            "q": "写一个诗歌，关于冬天",
            "a": [
                "冬夜寂静冷，",
                 "云在天边飘，", "冰封白雪上， ", "寒冷像一场雪。",
                 " ",
                 "雪花融化成冰，",
                 "像那雪花飘洒，",
                 "在寒冷的冬天，",
                 "感受春天的喜悦。",
                 " 冬日里，",
                 "风雪渐消，",
                 "一片寂静，",
                 "把快乐和温暖带回家。"
            ]
        }
        ]
    }



## 6.生成训练record

    python data_utils.py
    
    注:
    num_process_worker 为多进程制作数据 ， 如果数据量较大 ， 适当调大至cpu数量
    dataHelper.make_dataset_with_args(data_args.train_file,mixed_data=False, shuffle=True,mode='train',num_process_worker=0)


## 7.推理
    # infer.py 推理预训练模型
    # infer_finetuning.py 推理微调模型
    # infer_lora_finetuning.py 推理微调模型
     python infer.py



   

![inference](1.png)

## 8.训练
    支持以下微调方式 
- 冻结 N 层 修改models.py global_num_layers_freeze    
- lora
- 正常微调
```text
    训练精度 可以修改 config.json precision 16 32
    如果优化器不是lion ,可以尝试将config/config.json precision 改为32 ，然后修改 train.py Trainner 添加参数 precision=16 或者precision='bf16'
    python train.py
```

```text
多机多卡训练 例子 3个机器 每个机器 4个卡
修改train.py Trainer num_nodes = 3
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=0 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=1 python train.py 
MASTER_ADDR=10.0.0.1 MASTER_PORT=6667 WORLD_SIZE=12 NODE_RANK=2 python train.py 
```


### 是否开启lora finetuning
    with_lora


### int高效训练方式
   lora int8  修改modes.py load_in_8bit = True  修改data_utils.py 对应with_lora=True 
   注意 lora int8 多卡训练  CUDA_VISIBLE_DEVICES=显卡数量 和 data_utils.py devices 保持一致，否则容易出故障 ， 建议使用 CUDA_VISIBLE_DEVICES=显卡数量 python train.py

## 9.是否开启deepspeed
    启动则将data_utils.py  修改 enable_deepspeed 
    lora 模式暂时不支持deepspeed



## 10. Reference
    https://huggingface.co/fnlp
