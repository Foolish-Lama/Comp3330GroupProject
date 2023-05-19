
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import torchvision



class NaturalScenes():

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(144, 144)),
        transforms. RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    eval_transforms = transforms.Compose([
        transforms.Resize(size=(144, 144)), #some of the images are not 150x150
        transforms.ToTensor()
    ])

    def __init__(self, path):
        self.path = path
        
        full_data = torchvision.datasets.ImageFolder(self.path, transform=self.eval_transforms)

        train_size = int(len(full_data)*0.7)
        valid_size = int((len(full_data) - train_size) * 0.15)
        test_size = int(len(full_data) - train_size - valid_size)
        train_data, valid_data, test_data = random_split(full_data, [train_size, valid_size, test_size])
        #[full_data[x] for x in range(1400)]

        self.train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=len(valid_data), shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=True)
        self.data = [self.train_loader, self.valid_loader, self.test_loader]