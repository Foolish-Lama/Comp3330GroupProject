from torch import nn, optim

from naturalScenesData import NaturalScenes
from Model import Model

data = NaturalScenes()

#data.train_loader, data.valid_loader, data.test_loader


module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    ),
])

model = Model(1, module_list)
model.optimizer = optim.Adam(model.parameters(), lr=0.01)
model.loss_fn = nn.NLLLoss()

model.test_model()