
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder



class NaturalScenes():

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(scale=(0.6, 1.0), size=(144, 144)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    
    eval_transforms = transforms.Compose([
        transforms.Resize(size=(144, 144)), #some of the images are not 150x150
        transforms.ToTensor()
    ])

    def __init__(self, train_data_path, eval_data_path):
        self.train_data = ImageFolder(train_data_path, transform=self.train_transforms)
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=64, shuffle=True)

        eval_data = ImageFolder(eval_data_path, transform=self.eval_transforms)
        validation_size = int(len(eval_data)*0.5)
        test_size = int(len(eval_data) - validation_size)
        self.validation_data, self.test_data = random_split(eval_data, [validation_size, test_size])
        self.valid_loader = DataLoader(dataset=self.validation_data, batch_size=len(self.validation_data), shuffle=True)
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=len(self.test_data), shuffle=True)
        
        self.loaders = [self.train_loader, self.valid_loader, self.test_loader]

class NaturalScenesSubset(NaturalScenes):
    def __init__(self, train_data_path, eval_data_path):
        super().__init__(train_data_path, eval_data_path)
        self.train_data = Subset(self.train_data, [i for i, _ in enumerate(self.train_data) if i % 5 == 0])
        self.train_loader = DataLoader(dataset=self.train_data, batch_size=64, shuffle=True)

        self.validation_data = Subset(self.validation_data, [i for i, _ in enumerate(self.validation_data) if i % 5 == 0])
        self.valid_loader = DataLoader(dataset=self.validation_data, batch_size=len(self.validation_data), shuffle=True)
        
        self.test_data = Subset(self.test_data, [i for i, _ in enumerate(self.test_data) if i % 5 == 0])
        self.test_loader = DataLoader(dataset=self.test_data, batch_size=len(self.test_data), shuffle=True)
        
        self.loaders = [self.train_loader, self.valid_loader, self.test_loader]
