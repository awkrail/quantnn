from torchvision import transforms, datasets


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_set = datasets.MNIST("../../pytorch/data", train=True, download=False, transform=transform)
    calibration_num = 1000

    data_str_list = []
    for i in range(calibration_num):
        data = train_set[i][0].flatten().tolist()
        data_str = ','.join([str(x) for x in data])
        data_str_list.append("{{ {} }}".format(data_str))

    calibration_data_str = ',\n'.join(data_str_list)
    calibration_data_str = 'const std::vector<std::vector<float>> calibration_data = {{ {} }};\n'.format(calibration_data_str)

    with open('./calibration_data.h', 'w') as f:
        f.write(calibration_data_str)


if __name__ == "__main__":
    main()
