# This script converts our checkpoint to the form that gpt-fast can digest.
# Before running this script, first download our checkpoint. e.g.
#  s3://anyscale-staging-data-cld-kvedzwag2qa8i5bjxuevf5i7/org_7c1Kalm9WcX2bNIjW53GUT/cld_kvedZWag2qA8i5BjxUevf5i7/artifact_storage/xwjiang/llmforge-finetuning/spec_80m/TorchTrainer_2023-12-01_20-09-22/TorchTrainer_92c2f_00000_0_2023-12-01_20-09-22/checkpoint_000001 /mnt/local_storage/ckpt`
# After running this script, one should also copy tokenizer model from 7b to this directory (`/mnt/local_storage/ckpt/`).
# This finishes the preparation before running `python generate.py ...`

import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch

from transformers import AutoModelForCausalLM

weight_map = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
    "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
    "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    'model.layers.{}.self_attn.rotary_emb.inv_freq': None,
    'model.layers.{}.mlp.gate_proj.weight': 'layers.{}.feed_forward.w1.weight',
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}

def permute(w, n_head):
    dim = config.dim
    return (
        w.view(n_head, 2, config.head_dim // 2, dim)
        .transpose(1, 2)
        .reshape(config.head_dim * n_head, dim)
    )

model = AutoModelForCausalLM.from_pretrained("/mnt/local_storage/ckpt/", torch_dtype=torch.bfloat16)
merged_result = model.state_dict()

from model import ModelArgs

config = ModelArgs(n_layer=4, n_head=12, dim=768, intermediate_size=3072)

final_result = {}
for key, value in merged_result.items():
    if "layers" in key:
        abstract_key = re.sub(r'(\d+)', '{}', key)
        layer_num = re.search(r'\d+', key).group(0)
        new_key = weight_map[abstract_key]
        if new_key is None:
            continue
        new_key = new_key.format(layer_num)
    else:
        new_key = weight_map[key]

    final_result[new_key] = value

for key in tuple(final_result.keys()):
    if "wq" in key:
        q = final_result[key]
        k = final_result[key.replace("wq", "wk")]
        v = final_result[key.replace("wq", "wv")]
        q = permute(q, config.n_head)
        k = permute(k, config.n_local_heads)
        final_result[key.replace("wq", "wqkv")] = torch.cat([q, k, v])
        del final_result[key]
        del final_result[key.replace("wq", "wk")]
        del final_result[key.replace("wq", "wv")]

torch.save(final_result, "/mnt/local_storage/ckpt/model_new.pth")

