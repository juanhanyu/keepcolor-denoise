from pathlib import Path
import sys
import torch
import onnx
import numpy as np
import onnxruntime as ort
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'skills/keepcolor-denoise/scripts'))
from denoise import network

target = Path(__file__).resolve().parents[1] / 'public/drunet.onnx'
target.parent.mkdir(parents=True, exist_ok=True)
model = network('drunet', 'cpu').model.eval()
sample = torch.rand(1, 4, 128, 128)
torch.onnx.export(model, sample, str(target), input_names=['input'], output_names=['output'],
                  dynamic_axes={'input': {2: 'height', 3: 'width'}, 'output': {2: 'height', 3: 'width'}},
                  opset_version=17, dynamo=False)
onnx.checker.check_model(str(target))
session = ort.InferenceSession(str(target), providers=['CPUExecutionProvider'])
for height, width in [(8, 8), (24, 40)]:
    x = torch.rand(1, 4, height, width)
    with torch.no_grad():
        expected = model(x).numpy()
    actual = session.run(None, {'input': x.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=2e-4, atol=2e-5)
print('ONNX and PyTorch numerical comparison passed')
print(target)
