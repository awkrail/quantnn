# Quantnn
Quantization + DNN Inference engine from scratch to understand quantization techniques deeply.
The training code is written in Python+Pytorch and the inference one is written in C++.
The objective is to quantize & run MobileNetV3.

## MNIST + MLP

### 1. Train simple MLP on MNIST
Run `train_fc.py` and then run `./build/mnist_fc_float32`.
```
cd pytorch
python train_fc.py
cd ..
./build/mlp_float32
```

### 2. Int8 MLP Dynamic Quantization
The weights are saved as `int8_t`, but during the computation process, `int32_t` and `float32` are used to calculate scales/zero-points in the middle layers and activation functions; so, technically, the comsumed runtime memory is same as the original one.
```
./build/mlp_quantize_weight # convert float32 weights into INT8
./build/mlp_dynamic_quantization
```

### 3. Int8 MLP Static Quantization
The weights are saved as `int8_t` and scales/zero-points of middle layers are also saved.
Hence, the comsumed runtime memory should be about 1/4 to the original fp32.
```
./build/mlp_calibration
./build/mlp_static_quantization
```

## ConvNet

### 4. Train ConvNet on MNIST
```
cd pytorch
python train_convnet.py
cd ..
./build/conv_float32
```

### 5. Int8 MLP Dynamic Quantization
```
./build/conv_quantize_weight
./build/conv_dynamic_quantization
```

### 6. Int8 MLP Static Quantization
```
./build/conv_calibration
./build/conv_static_quantization
```
