import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from mobileone import mobileone

def load_labels(filename):
    with open(filename, 'r') as f:
        idx_to_labels = [line.strip() for line in f.readlines()]
    return idx_to_labels

def main():
    model = mobileone(inference_mode=True)
    checkpoint = torch.load('./mobileone_s0.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )
    ])

    img = Image.open('../misc/images/apple.png').convert("RGB")
    idx_to_labels = load_labels('../misc/imagenet_classes.txt')
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
    pred = torch.topk(output, dim=1, k=5)
    pred_indices = pred.indices[0]

    for i, pred_idx in enumerate(pred_indices):
        print(f"Top-{i+1} Predicted class index: {idx_to_labels[pred_idx]}")


if __name__ == "__main__":
    main()
