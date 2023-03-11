
import numpy as np
import time as t
import onnxruntime as ox
VIDEO_SIZE = (1920, 1080)
SCORE_THRESHOLD = 0.15
MODEL_DIR="./"
MODEL_NAME='car.onnx'

MOCK_INPUT=[[0,1,0,0],[0,0,0,0]]
MOCK_CAR_INPUT=np.ones((1,96,96,5)).astype(np.float32)

#init onnx runtime
model=MODEL_DIR+MODEL_NAME
providers = [
('CUDAExecutionProvider', {
    'device_id': 0,
    'arena_extend_strategy': 'kNextPowerOfTwo',
    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,
    'cudnn_conv_algo_search': 'EXHAUSTIVE',
    'do_copy_in_default_stream': True,
}),
'CPUExecutionProvider',
]
session = ox.InferenceSession(model,providers=providers)
input_name = session.get_inputs()[0].name
print("input name", input_name)
input_shape = session.get_inputs()[0].shape
print("input shape", input_shape)
input_type = session.get_inputs()[0].type
print("input type", input_type)
output_name = session.get_outputs()[0].name
print("output name", output_name)
output_shape = session.get_outputs()[0].shape
print("output shape", output_shape)
output_type = session.get_outputs()[0].type
print("output type", output_type)

tic=t.time()
for i in range(0,100):
	outputs = session.run([output_name], {input_name:MOCK_CAR_INPUT})[0]
toc=t.time()

print(outputs,'fps=',1/(toc-tic)*100)


