import argparse
from pathlib import Path

from onnxruntime.quantization import QuantType, quantize_dynamic


def main():
    parser = argparse.ArgumentParser(description="Quantize English MeloTTS ONNX model (dynamic INT8)")
    parser.add_argument("--onnx_dir", type=str, default="onnx_models/tts_onnx_EN", help="Directory containing tts_EN ONNX model")
    parser.add_argument("--input", type=str, default=None, help="Input ONNX model path")
    parser.add_argument("--output", type=str, default=None, help="Output quantized ONNX model path")
    args = parser.parse_args()

    root = Path(args.onnx_dir)
    input_model = Path(args.input) if args.input else (root / "tts_EN.onnx")
    output_model = Path(args.output) if args.output else (root / "tts_EN_int8.onnx")
    output_model.parent.mkdir(parents=True, exist_ok=True)

    quantize_dynamic(
        model_input=str(input_model),
        model_output=str(output_model),
        weight_type=QuantType.QInt8,
        per_channel=True,
    )

    print(f"Quantized EN model saved to: {output_model}")


if __name__ == "__main__":
    main()
