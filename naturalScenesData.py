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
        transforms.Resize(size=(144, 144)),
        transforms.ToTensor()
    ])

    def __init__(self):
        full_data = torchvision.datasets.ImageFolder("NaturalScenes\seg_train", transform=self.eval_transforms)

        train_data, valid_data, test_data = random_split([full_data[x] for x in range(1400)], [1000, 200, 200])
        #
        #[full_data[x] for x in range(1400)]
        self.train_loader = DataLoader(dataset=train_data, batch_size=64, shuffle=True)
        self.valid_loader = DataLoader(dataset=valid_data, batch_size=len(valid_data), shuffle=True)
        self.test_loader = DataLoader(dataset=test_data, batch_size=len(test_data), shuffle=True)
        self.data = [self.train_loader, self.valid_loader, self.test_loader]