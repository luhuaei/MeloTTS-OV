import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    parser = argparse.ArgumentParser(description="Quantize MeloTTS ONNX model (dynamic INT8)")
    parser.add_argument("--input", type=str, required=True, help="Input ONNX model path")
    parser.add_argument("--output", type=str, required=True, help="Output quantized ONNX model path")
    parser.add_argument(
        "--weight_type",
        type=str,
        choices=["qint8", "quint8"],
        default="qint8",
        help="Quantized weight type",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_path),
        model_output=str(output_path),
        weight_type=QuantType.QInt8 if args.weight_type == "qint8" else QuantType.QUInt8,
        per_channel=True,
    )

    print(f"Quantized model saved to: {output_path}")


if __name__ == "__main__":
    main()
