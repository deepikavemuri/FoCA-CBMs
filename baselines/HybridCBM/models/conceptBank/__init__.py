# -*- coding: utf-8 -*- 
"""
@Time : 2024/9/23 20:09 
@Author :   liuyang 
@github :   https://github.com/ly1998117/MMCBM
@Contact :  liu.yang.mine@gmail.com
@File :     __init__.py.py 
"""
from .hybrid_bank import HybridConceptBank, HybridAttnConceptBank


def get_concept_bank_fromconfig(config):
    return HybridConceptBank(
        exp_root=config.exp_root,
        data_root=config.data_root,
        # concept
        num_static_concept=config.num_static_concept,
        num_dynamic_concept=config.num_dynamic_concept,
        concept_select_fn=config.concept_select_fn,
        submodular_weights=config.submodular_weights,

        # clip
        clip_model=config.clip_model,
        translator_path=config.translator_path,
    )


def get_attention_bank_fromconfig(config):
    return HybridAttnConceptBank(
        exp_root=config.exp_root,
        data_root=config.data_root,
        # concept
        num_static_concept=config.num_static_concept,
        num_dynamic_concept=config.num_dynamic_concept,
        concept_select_fn=config.concept_select_fn,
        submodular_weights=config.submodular_weights,

        # clip
        clip_model=config.clip_model,
        translator_path=config.translator_path,
    )
