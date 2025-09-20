import os
import numpy as np
import tempfile, zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torchvision
    import torchaudio
except:
    pass

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear_0 = nn.Linear(bias=True, in_features=784, out_features=128)
        self.linear_1 = nn.Linear(bias=True, in_features=128, out_features=10)

        archive = zipfile.ZipFile('mnist_fc.pnnx.bin', 'r')
        self.linear_0.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_0.bias', (128), 'float32')
        self.linear_0.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_0.weight', (128,784), 'float32')
        self.linear_1.bias = self.load_pnnx_bin_as_parameter(archive, 'linear_1.bias', (10), 'float32')
        self.linear_1.weight = self.load_pnnx_bin_as_parameter(archive, 'linear_1.weight', (10,128), 'float32')
        archive.close()

    def load_pnnx_bin_as_parameter(self, archive, key, shape, dtype, requires_grad=True):
        return nn.Parameter(self.load_pnnx_bin_as_tensor(archive, key, shape, dtype), requires_grad)

    def load_pnnx_bin_as_tensor(self, archive, key, shape, dtype):
        fd, tmppath = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as tmpf, archive.open(key) as keyfile:
            tmpf.write(keyfile.read())
        m = np.memmap(tmppath, dtype=dtype, mode='r', shape=shape).copy()
        os.remove(tmppath)
        return torch.from_numpy(m)

    def forward(self, v_0):
        v_1 = self.linear_0(v_0)
        v_2 = F.relu(input=v_1)
        v_3 = self.linear_1(v_2)
        return v_3

def export_torchscript():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 784, dtype=torch.float)

    mod = torch.jit.trace(net, v_0)
    mod.save("mnist_fc_pnnx.py.pt")

def export_onnx():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 784, dtype=torch.float)

    torch.onnx.export(net, v_0, "mnist_fc_pnnx.py.onnx", export_params=True, operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK, opset_version=13, input_names=['in0'], output_names=['out0'])

@torch.no_grad()
def test_inference():
    net = Model()
    net.float()
    net.eval()

    torch.manual_seed(0)
    v_0 = torch.rand(1, 784, dtype=torch.float)

    return net(v_0)

if __name__ == "__main__":
    print(test_inference())
