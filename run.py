import argparse
import json
import os
import time

import tensorrt as trt
import tensorrt_llm
import torch
from build import get_engine_name
from datasets import concatenate_datasets, load_dataset
from tensorrt_llm import logger
from tensorrt_llm.runtime import Session, TensorInfo
from torch.nn.functional import cosine_similarity
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError("%s is not supported" % dtype)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_level", type=str, default="info")
    parser.add_argument("--engine_dir", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="shreyasharma/sentences_truth")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--run_hf", action="store_true")
    parser.add_argument("--run_trtllm", action="store_true")
    parser.add_argument("--remove_columns", default="labels", type=str)
    parser.add_argument("--target_column", default="sentences", type=str)
    return parser.parse_args()


def main():
    args = parse_arguments()
    tensorrt_llm.logger.set_level(args.log_level)

    config_path = os.path.join(args.engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    dtype = config["builder_config"]["precision"]
    world_size = config["builder_config"]["tensor_parallel"]
    assert (
        world_size == tensorrt_llm.mpi_world_size()
    ), f"Engine world size ({world_size}) != Runtime world size ({tensorrt_llm.mpi_world_size()})"

    model_name = config["builder_config"]["name"]
    runtime_rank = tensorrt_llm.mpi_rank() if world_size > 1 else 0

    runtime_mapping = tensorrt_llm.Mapping(world_size, runtime_rank, tp_size=world_size)
    torch.cuda.set_device(runtime_rank % runtime_mapping.gpus_per_node)

    serialize_path = get_engine_name(model_name, dtype, world_size, runtime_rank)
    serialize_path = os.path.join(args.engine_dir, serialize_path)

    stream = torch.cuda.current_stream().cuda_stream
    logger.info(f"Loading engine from {serialize_path}")
    with open(serialize_path, "rb") as f:
        engine_buffer = f.read()
    logger.info(f"Creating session from engine")
    session = Session.from_serialized_engine(engine_buffer)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if args.run_hf:
        hf_model = (
            AutoModel.from_pretrained(
                model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
            )
            .eval()
            .cuda()
        )

    dataset = load_dataset(args.dataset)
    dataset = concatenate_datasets(list(dataset.values())).remove_columns(
        args.remove_columns
    )

    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    total_time_trtllm = 0
    total_trtllm = 0

    total_time_hf = 0
    total_hf = 0

    for batch in tqdm(dataloader, unit_scale=args.batch_size, unit=" samples"):
        encoded_input = tokenizer(
            batch[args.target_column],
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoded_input["input_ids"].cuda()
        input_lengths = torch.tensor(
            [seq.shape[-1] for seq in encoded_input["input_ids"]]
        ).cuda()
        token_type_ids = encoded_input["token_type_ids"].cuda()

        inputs = {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_type_ids": token_type_ids,
        }

        output_info = session.infer_shapes(
            [
                TensorInfo("input_ids", trt.DataType.INT32, input_ids.shape),
                TensorInfo("input_lengths", trt.DataType.INT32, input_lengths.shape),
                TensorInfo("token_type_ids", trt.DataType.INT32, token_type_ids.shape),
            ]
        )

        outputs = {
            t.name: torch.empty(
                tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device="cuda"
            )
            for t in output_info
        }

        output_name = "hidden_states"
        assert (
            output_name in outputs
        ), f"{output_name} not found in outputs, check if build.py set the name correctly"

        if args.run_trtllm:
            start = time.time()
            ok = session.run(inputs, outputs, stream)
            total_time_trtllm += time.time() - start
            total_trtllm += 1
            assert ok, "Runtime execution failed"
            torch.cuda.synchronize()
            res = outputs[output_name]
            trtllm_embeddings = res[:, 0, :]  # perform cls pooling

        if args.run_hf:
            with torch.inference_mode():
                start = time.time()
                model_output = hf_model(
                    input_ids=input_ids, token_type_ids=token_type_ids
                )
                total_time_hf += time.time() - start
                total_hf += 1
                hf_embeddings = model_output[0][:, 0, :]  # perform cls pooling

        if args.run_hf and args.run_trtllm:
            print(
                f"Average Cosine Distance: {(1 - cosine_similarity(trtllm_embeddings, hf_embeddings).mean().item()):.2f}"
            )

            print(trtllm_embeddings)
            print(hf_embeddings)

    if args.run_hf:
        print(total_time_hf / total_hf)
    if args.run_trtllm:
        print(total_time_trtllm / total_trtllm)


if __name__ == "__main__":
    main()
