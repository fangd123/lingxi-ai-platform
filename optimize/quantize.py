import onnxruntime
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx
import os
from loguru import logger
import fire
def quantize_onnx_model(onnx_model_path, quantized_model_path):
    quantize_dynamic(onnx_model_path,
                     quantized_model_path,
                     weight_type=QuantType.QInt8)

    logger.info(f"quantized model saved to:{quantized_model_path}")

if __name__ == "__main__":
    fire.Fire(quantize_onnx_model)
    
