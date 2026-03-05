import argparse

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer


def pad_to_fixed_len(input_ids: np.ndarray, fixed_len: int) -> np.ndarray:
    if input_ids.shape[1] == fixed_len:
        return input_ids
    if input_ids.shape[1] > fixed_len:
        return input_ids[:, :fixed_len]

    out = np.zeros((input_ids.shape[0], fixed_len), dtype=input_ids.dtype)
    out[:, : input_ids.shape[1]] = input_ids
    return out


def main():
    parser = argparse.ArgumentParser(description="Validate fixed-length BERT ONNX input for edge devices")
    parser.add_argument("--model", type=str, required=True, help="BERT ONNX model path")
    parser.add_argument("--tokenizer", type=str, default="bert-base-multilingual-uncased", help="Tokenizer id/path")
    parser.add_argument("--seq_len", type=int, default=32, help="Fixed sequence length")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="ONNX Runtime provider")
    args = parser.parse_args()

    providers = ["CPUExecutionProvider"]
    if args.device in {"auto", "cuda"} and "CUDAExecutionProvider" in ort.get_available_providers():
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    session = ort.InferenceSession(args.model, providers=providers)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    text = "A buffer is a container for data that can be accessed from a device and the host."
    encoded = tokenizer(text, return_tensors="np")

    feed = {
        "input_ids": pad_to_fixed_len(encoded["input_ids"], args.seq_len).astype(np.int64),
        "token_type_ids": pad_to_fixed_len(encoded["token_type_ids"], args.seq_len).astype(np.int64),
        "attention_mask": pad_to_fixed_len(encoded["attention_mask"], args.seq_len).astype(np.int64),
    }

    output_name = session.get_outputs()[0].name
    output = session.run([output_name], feed)[0]
    print(f"Inference succeeded. output shape: {output.shape}")


if __name__ == "__main__":
    main()
