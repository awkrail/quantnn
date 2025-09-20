import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

class MnistFC(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(784, 128)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
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
        batch_size, _, h, w = x.shape
        x = x.view(batch_size, h * w)
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
            batch_size, _, h, w = x.shape
            x = x.view(batch_size, h * w)
            logits = model(x)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main():
    train_loader, test_loader = get_loaders()
    model = MnistFC()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(5):
        loss = train_one_epoch(model, train_loader, optimizer)
        acc = evaluate(model, test_loader)
        print(f"Epoch {epoch+1} loss={loss:.4f} acc={acc*100:.2f}%")


    # save weight/bias as C++ float array
    fc1_weights = model.fc1.weight.flatten().tolist()
    fc2_weights = model.fc2.weight.flatten().tolist()

    fc1_bias = model.fc1.bias.tolist()
    fc2_bias = model.fc2.bias.tolist()
    
    fc1_weight_str = ','.join([str(x) for x in fc1_weights])
    fc2_weight_str = ','.join([str(x) for x in fc2_weights])
    fc1_bias_str = ','.join([str(x) for x in fc1_bias])
    fc2_bias_str = ','.join([str(x) for x in fc2_bias])

    weight_const_str = "const float fc1_weight [] = {{ {} }};\nconst float fc1_bias [] = {{ {} }};\nconst float fc2_weight [] = {{ {} }};\nconst float fc2_bias [] = {{ {} }};".format(fc1_weight_str, fc1_bias_str, fc2_weight_str, fc2_bias_str)
    with open('../src/mnist_fc.h', 'w') as f:
        f.write(weight_const_str)

    # save sample data as data
    data = test_loader.dataset[0][0].flatten().tolist()
    label = test_loader.dataset[0][1]

    data_str = ','.join([str(x) for x in test_loader.dataset[0][0].flatten().tolist()])
    data_const_str = "const float data [] = {{ {} }};\n".format(data_str)
    with open('../src/data_{}.h'.format(label), 'w') as f:
        f.write(data_const_str)


if __name__ == "__main__":
    main()
