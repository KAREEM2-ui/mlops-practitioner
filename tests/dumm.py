

import onnxruntime

from proj_1.model import ONNXModel



model = ONNXModel(model_path="model.onnx")

print("Model loaded successfully.")


# after loading the exported onnx model:
session = onnxruntime.InferenceSession("model.onnx")
for out in session.get_outputs():
    print(out.name, out.shape)