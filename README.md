# Quantnn
Quantization + DNN Inference engine from scratch to understand quantization techniques deeply.
The training code is written in Python+Pytorch and the inference one is written in C++.
The objective is to quantize & run MobileNetV3.

## MNIST

### Level.1: Train simple MLP on MNIST / inference w/ fp32
Run `train_fc.py` and then run `./build/mnist_fc_float32`.
```
cd pytorch
python train_fc.py
cd ..
./build/mnist_fc_float32
```

### Level.2: Int8 Dynamic Quantization
The weights are saved as `int8_t`, but during the computation process, `int32_t` and `float32` are used to calculate scales/zero-points in the middle layers and activation functions; so, technically, the comsumed runtime memory is same as the original one.
```
./build/quantize_weight # convert float32 weights into INT8
./build/dynamic_quantization
```

### Level.3: Static Quantization (INT8)
The weights are saved as `int8_t` and scales/zero-points of middle layers are also saved.
Hence, the comsumed runtime memory should be about 1/4 to the original fp32.
```
./build/calibration
./build/static_quantization
```

### Level.4: ConvNet (FP32)
Run `train_convnet.py` and then run `./build/convnet_float32`
```
cd pytorch
python train_convnet.py
cd ..
./build/convnet_float32
```
We used `im2col` techniques.


### Level.4: Quantization-aware Training

**WIP**

## ConvNet

**WIP**
