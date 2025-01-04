import os
os.environ['CUDA_MODULE_LOADING'] = 'LAZY'
import onnx
import onnxruntime as rt
import numpy as np
from PIL import Image
import time

# Load the ResNet50 model
start_time = time.time()
model = onnx.load("resnet50-v2-7.onnx")
providers = [
    ('TensorrtExecutionProvider', {
        'device_id': 0,
        'trt_dla_enable': True,
        'trt_max_workspace_size': 4294967296,
        'trt_fp16_enable': True,
    })
]
sess = rt.InferenceSession(model.SerializeToString(), providers=providers)

# Load an image
img = Image.open("dog.jpg")

# Preprocess the image
img = img.resize((224, 224))
img = np.array(img).transpose((2, 0, 1))
img = np.expand_dims(img, axis=0).astype(np.float32)
img = (img - 127.5) / 128

# Run inference on the image
input_name = sess.get_inputs()[0].name

# Run the model for 100 times and get the average time
time_costs = []
for _ in range(100):
    start = time.time()
    outputs = sess.run(None, {input_name: img})
    time_costs.append(time.time() - start)

avg_time_cost = sum(time_costs) / len(time_costs)
print(f"Average time cost: {avg_time_cost}")

# Get the top-k predictions from the model
predictions = outputs[0].flatten()
top_k = predictions.argsort()[-5:][::-1]

# Load ImageNet classes
with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

# Print top-k predictions
for i in top_k:
    print(f"Class: {classes[i]}, Score: {predictions[i]}")