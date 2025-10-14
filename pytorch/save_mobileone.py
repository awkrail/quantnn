import torch
from mobileone import mobileone

def transform_param_to_str(param):
    param_list = []
    for x in param:
        x = str(x.item())[:6] + 'f'
        param_list.append(x)
    param_list_str = ', '.join(param_list)
    return param_list_str


def main():
    model = mobileone(inference_mode=True)
    checkpoint = torch.load('./mobileone_s0.pth.tar', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()

    param_str_list = []
    for name, param in model.named_parameters():
        param_str = transform_param_to_str(param.flatten())
        param_str = "const std::vector<float> {} = {{ {} }};\n".format(name.replace('.', '_'), param_str)
        param_str_list.append(param_str)

    with open('../src/mobileone/mobileone.h', 'w') as f:
        for param_str in param_str_list:
            f.write(param_str + '\n')

if __name__ == "__main__":
    main()
