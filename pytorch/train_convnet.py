import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class MnistConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 5, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu1 = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(5 * 28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        B, _, _, _ = x.shape
        x = x.view(B, -1)
        x = self.relu1(self.fc1(x))
        x = self.fc2(x)
        return x


def get_loaders(batch_size = 512):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("./data", train=True, download=True, transform=tfm)
    test_set = datasets.MNIST("./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=512, shuffle=False, num_workers=2)
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer):
    model.train()
    crit = nn.CrossEntropyLoss()
    running = 0.0
    for x, y in loader:
        optimizer.zero_grad()
        logits = model(x)
        loss = crit(logits, y)
        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
    return running / len(loader.dataset)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    train_loader, test_loader = get_loaders()
    model = MnistConvNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5): # 5
        loss = train_one_epoch(model, train_loader, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1} loss={loss:.4f} acc={acc*100:.2f}%")

    # save weight/bias as C++ float array
    conv1_weight_str = ','.join([str(x) for x in model.conv1.weight.flatten().tolist()])
    fc1_weight_str = ','.join([str(x) for x in model.fc1.weight.flatten().tolist()])
    fc2_weight_str = ','.join([str(x) for x in model.fc2.weight.flatten().tolist()])
    conv1_bias_str = ','.join([str(x) for x in model.conv1.bias.flatten().tolist()])
    fc1_bias_str = ','.join([str(x) for x in model.fc1.bias.flatten().tolist()])
    fc2_bias_str = ','.join([str(x) for x in model.fc2.bias.flatten().tolist()])

    weight_const_str = (
        "const std::vector<float> conv1_weight = {{ {} }};\n"
        "const std::vector<float> conv1_bias = {{ {} }};\n"
        "const std::vector<float> fc1_weight = {{ {} }};\n"
        "const std::vector<float> fc1_bias = {{ {} }};\n"
        "const std::vector<float> fc2_weight = {{ {} }};\n"
        "const std::vector<float> fc2_bias = {{ {} }};"
    ).format(conv1_weight_str, conv1_bias_str, 
             fc1_weight_str, fc1_bias_str,
             fc2_weight_str, fc2_bias_str)

    with open('../src/fp32/mnist_conv.h', 'w') as f:
        f.write(weight_const_str)


if __name__ == "__main__":
    main()
