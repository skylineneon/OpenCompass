# configs/llama7b.py
from mmengine.config import read_base

with read_base():
    # 直接从预设数据集配置中读取需要的数据集配置
    from .datasets.piqa.piqa_ppl import piqa_datasets

# 将需要评测的数据集拼接成 datasets 字段
datasets = [*piqa_datasets]

# 使用 HuggingFaceCausalLM 评测 HuggingFace 中 AutoModelForCausalLM 支持的模型
from opencompass.models import HuggingFaceCausalLM
mode_id = "/root/workspace/modelscope/cache/qwen/Qwen2___5-0___5B-Instruct"
models = [
    dict(
        type=HuggingFaceCausalLM,
        # 以下参数为 HuggingFaceCausalLM 的初始化参数
        path=mode_id,
        tokenizer_path=mode_id,
        tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
        max_seq_len=2048,
        # 以下参数为各类模型都必须设定的参数，非 HuggingFaceCausalLM 的初始化参数
        abbr='qwen2.5-0.5b',            # 模型简称，用于结果展示
        max_out_len=256,            # 最长生成 token 数
        batch_size=128,              # 批次大小
        run_cfg=dict(num_gpus=1),   # 运行配置，用于指定资源需求
    )
]