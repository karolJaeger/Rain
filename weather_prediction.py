import os

import torch
from torch.utils.data import DataLoader
import model as M


def get_path_weight(day):
    folder_path = f"weights/main/best/"
    file_list = os.listdir(folder_path)
    sorted_files = sorted(file_list, key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    newest_file_name = sorted_files[0]
    newest_file_path = os.path.join(folder_path, newest_file_name)

    return newest_file_path


if __name__ == "__main__":
    # model = M.Model(patch_size=M.patch_size, embed_dim=M.embed_dim, num_heads=M.num_heads, attention_layer=M.attention_layer)
    model = M.Model()
    for day in range(1):
        data = M.WeatherData()
        # model = torch.load(f'weights/model/model.pt')
        # model.load_state_dict(torch.load('weights/main/best/15.pt'))
        model.load_state_dict(torch.load(get_path_weight(day=day)))
        model.to('cuda')
        loader = DataLoader(data)
        for n, (pos, img, date) in enumerate(loader):
            img = img.to('cuda')
            pred = model(pos, img)

            if date[0][11:13] == '00':
                print(f'  \t {int(pred.item() * 100)}')
            else:
                print(f'{date[0]} \t {int(pred.item()*100)}')


