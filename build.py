import os
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import tensorrt_llm
import torch
from tensorrt_llm.builder import Builder
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.network import net_guard
from tensorrt_llm.plugin.plugin import ContextFMHAType
from transformers import AutoModel, BertConfig, BertPreTrainedModel


def parse_arguments():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--world_size", type=int, default=1, help="Tensor parallelism size"
    )
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument(
        "--dtype", type=str, default="float16", choices=["float16", "float32"]
    )
    parser.add_argument("--timing_cache", type=str, default="model.cache")
    parser.add_argument(
        "--profiling_verbosity",
        type=str,
        default="layer_names_only",
        choices=["layer_names_only", "detailed", "none"],
        help="The profiling verbosity for the generated TRT engine. Set to detailed can inspect tactic choices and kernel parameters.",
    )
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--max_batch_size", type=int, default=256)
    parser.add_argument("--max_input_len", type=int, default=512)
    parser.add_argument("--gpus_per_node", type=int, default=1)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument(
        "--use_bert_attention_plugin",
        nargs="?",
        const="float16",
        type=str,
        default=False,
        choices=["float16", "float32"],
    )
    parser.add_argument(
        "--use_gemm_plugin",
        nargs="?",
        const="float16",
        type=str,
        default=False,
        choices=["float16", "float32"],
    )
    parser.add_argument("--enable_qk_half_accum", default=False, action="store_true")
    parser.add_argument("--enable_context_fmha", default=False, action="store_true")
    parser.add_argument(
        "--enable_context_fmha_fp32_acc", default=False, action="store_true"
    )
    parser.add_argument("--model", type=str, help="Model id")
    return parser.parse_args()


def get_engine_name(model, dtype, tp_size, rank):
    return "{}_{}_tp{}_rank{}.engine".format(
        model.replace("/", "--"), dtype, tp_size, rank
    )


def extract_layer_idx(name):
    ss = name.split(".")
    for s in ss:
        if s.isdigit():
            return s
    return None


def split(v, tp_size, idx, dim=0):
    if tp_size == 1:
        return v
    if len(v.shape) == 1:
        return np.ascontiguousarray(np.split(v, tp_size)[idx].copy())
    elif len(v.shape) == 2:
        return np.ascontiguousarray(np.split(v, tp_size, axis=dim)[idx].copy())
    return None


def load_from_hf_model(
    tensorrt_llm_model: tensorrt_llm.models.BertModel,
    hf_model: BertPreTrainedModel,
    hf_model_config: BertConfig,
    rank=0,
    tensor_parallel=1,
    fp16=False,
):
    qkv_weight = [[None, None, None] for _ in range(hf_model_config.num_hidden_layers)]

    qkv_bias = [[None, None, None] for _ in range(hf_model_config.num_hidden_layers)]

    torch_dtype = torch.float16 if fp16 else torch.float32
    for k, v in hf_model.state_dict().items():
        v = v.to(torch_dtype).cpu().numpy()
        if "embeddings.word_embeddings.weight" in k:
            tensorrt_llm_model.embedding.vocab_embedding.weight.value = v
        elif "embeddings.position_embeddings.weight" in k:
            tensorrt_llm_model.embedding.position_embedding.weight.value = v
        elif "embeddings.token_type_embeddings.weight" in k:
            tensorrt_llm_model.embedding.token_embedding.weight.value = v
        elif "embeddings.LayerNorm.weight" in k:
            tensorrt_llm_model.embedding.embedding_ln.weight.value = v
        elif "embeddings.LayerNorm.bias" in k:
            tensorrt_llm_model.embedding.embedding_ln.bias.value = v
        else:
            layer_idx = extract_layer_idx(k)
            if layer_idx is None:
                continue
            idx = int(layer_idx)
            if "attention.output.dense.weight" in k:
                tensorrt_llm_model.layers[idx].attention.dense.weight.value = split(
                    v, tensor_parallel, rank, dim=1
                )
            elif "attention.output.dense.bias" in k:
                tensorrt_llm_model.layers[idx].attention.dense.bias.value = v
            elif "attention.output.LayerNorm.weight" in k:
                tensorrt_llm_model.layers[idx].input_layernorm.weight.value = v
            elif "attention.output.LayerNorm.bias" in k:
                tensorrt_llm_model.layers[idx].input_layernorm.bias.value = v
            elif "intermediate.dense.weight" in k:
                tensorrt_llm_model.layers[idx].mlp.fc.weight.value = split(
                    v, tensor_parallel, rank
                )
            elif "intermediate.dense.bias" in k:
                tensorrt_llm_model.layers[idx].mlp.fc.bias.value = split(
                    v, tensor_parallel, rank
                )
            elif "output.dense.weight" in k:
                tensorrt_llm_model.layers[idx].mlp.proj.weight.value = split(
                    v, tensor_parallel, rank, dim=1
                )
            elif "output.dense.bias" in k:
                tensorrt_llm_model.layers[idx].mlp.proj.bias.value = v
            elif "output.LayerNorm.weight" in k:
                tensorrt_llm_model.layers[idx].post_layernorm.weight.value = v
            elif "output.LayerNorm.bias" in k:
                tensorrt_llm_model.layers[idx].post_layernorm.bias.value = v
            elif "attention.self.query.weight" in k:
                qkv_weight[idx][0] = v
            elif "attention.self.query.bias" in k:
                qkv_bias[idx][0] = v
            elif "attention.self.key.weight" in k:
                qkv_weight[idx][1] = v
            elif "attention.self.key.bias" in k:
                qkv_bias[idx][1] = v
            elif "attention.self.value.weight" in k:
                qkv_weight[idx][2] = v
            elif "attention.self.value.bias" in k:
                qkv_bias[idx][2] = v

    for i in range(hf_model_config.num_hidden_layers):
        tensorrt_llm_model.layers[i].attention.qkv.weight.value = split(
            np.concatenate(qkv_weight[i]), tensor_parallel, rank
        )
        tensorrt_llm_model.layers[i].attention.qkv.bias.value = split(
            np.concatenate(qkv_bias[i]), tensor_parallel, rank
        )


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    bs_range = [1, (args.max_batch_size + 1) // 2, args.max_batch_size]
    inlen_range = [1, (args.max_input_len + 1) // 2, args.max_input_len]
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.float32
    trt_dtype = trt.float16 if args.dtype == "float16" else trt.float32

    builder = Builder()
    builder_config = builder.create_builder_config(
        name=args.model,
        precision=args.dtype,
        timing_cache=args.timing_cache,
        profiling_verbosity=args.profiling_verbosity,
        tensor_parallel=args.world_size,  # TP only
        max_batch_size=args.max_batch_size,
        max_input_len=args.max_input_len,
    )

    hf_model = (
        AutoModel.from_pretrained(args.model, torch_dtype=torch_dtype)
        .cuda()
        .to(torch_dtype)
        .eval()
    )

    output_name = "hidden_states"

    tensorrt_llm_bert = tensorrt_llm.models.BertModel(
        num_layers=hf_model.config.num_hidden_layers,
        num_heads=hf_model.config.num_attention_heads,
        hidden_size=hf_model.config.hidden_size,
        vocab_size=hf_model.config.vocab_size,
        hidden_act=hf_model.config.hidden_act,
        max_position_embeddings=hf_model.config.max_position_embeddings,
        type_vocab_size=hf_model.config.type_vocab_size,
        pad_token_id=hf_model.config.pad_token_id,
        is_roberta=False,
        mapping=Mapping(
            world_size=args.world_size, rank=args.rank, tp_size=args.world_size
        ),  # TP only
        dtype=trt_dtype,
    )
    load_from_hf_model(
        tensorrt_llm_bert,
        hf_model,
        hf_model.config,
        rank=args.rank,
        tensor_parallel=args.world_size,
        fp16=(args.dtype == "float16"),
    )

    network = builder.create_network()
    network.plugin_config.to_legacy_setting()
    if args.use_bert_attention_plugin:
        network.plugin_config.set_bert_attention_plugin(
            dtype=args.use_bert_attention_plugin
        )
    if args.use_gemm_plugin:
        network.plugin_config.set_gemm_plugin(dtype=args.use_gemm_plugin)
    if args.enable_qk_half_accum:
        network.plugin_config.enable_qk_half_accum()
    assert not (args.enable_context_fmha and args.enable_context_fmha_fp32_acc)
    if args.enable_context_fmha:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled)
    if args.enable_context_fmha_fp32_acc:
        network.plugin_config.set_context_fmha(ContextFMHAType.enabled_with_fp32_acc)
    if args.world_size > 1:
        network.plugin_config.set_nccl_plugin(args.dtype)
    with net_guard(network):
        # Prepare
        network.set_named_parameters(tensorrt_llm_bert.named_parameters())

        # Forward
        input_ids = tensorrt_llm.Tensor(
            name="input_ids",
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict(
                [("batch_size", [bs_range]), ("input_len", [inlen_range])]
            ),
        )

        # also called segment_ids
        token_type_ids = tensorrt_llm.Tensor(
            name="token_type_ids",
            dtype=trt.int32,
            shape=[-1, -1],
            dim_range=OrderedDict(
                [("batch_size", [bs_range]), ("input_len", [inlen_range])]
            ),
        )

        input_lengths = tensorrt_llm.Tensor(
            name="input_lengths",
            dtype=trt.int32,
            shape=[-1],
            dim_range=OrderedDict([("batch_size", [bs_range])]),
        )

        output = tensorrt_llm_bert(
            input_ids=input_ids,
            input_lengths=input_lengths,
            token_type_ids=token_type_ids,
        )

        output_dtype = trt.float16 if args.dtype == "float16" else trt.float32
        output.mark_output(output_name, output_dtype)

    # Network -> Engine
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, "Failed to build engine."
    engine_file = os.path.join(
        args.output_dir,
        get_engine_name(args.model, args.dtype, args.world_size, args.rank),
    )
    with open(engine_file, "wb") as f:
        f.write(engine)
    builder.save_config(builder_config, os.path.join(args.output_dir, "config.json"))


if __name__ == "__main__":
    main()
