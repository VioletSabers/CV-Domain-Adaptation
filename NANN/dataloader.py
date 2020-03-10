import os
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, ok_label=[], radio=1, reverse=False):
        folders = os.listdir(root_dir)
        self.data = []
        self.label = folders
        for i, folder in enumerate(folders):
            if len(ok_label) != 0 and folder not in ok_label:
                continue
            folders_list = os.listdir(os.path.join(root_dir, folder))
            if reverse == False:
                slice = folders_list[:int(radio * len(folders_list))]
            else:
                slice = folders_list[int(-radio * len(folders_list)):]
            for img in slice:
                path = os.path.join(root_dir, folder, img)
                self.data.append((path, int(folder)))
        self.transform = transform

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.data)

    def get_label(self, index):
        return self.label[index]

# ok_label = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet']
ok_label = []
def load_train_data(path, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])
         ]
    )
    data = MyDataset(root_dir=path, transform=transform, ok_label=ok_label)
    train_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return train_loader

def load_test_data(path, batch_size):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.5, 0.5, 0.5],
                              std=[0.5, 0.5, 0.5])
         ]
    )
    data = MyDataset(root_dir=path, transform=transform, ok_label=ok_label)
    test_loader = DataLoader(data, batch_size=batch_size, shuffle=True)
    return test_loader