
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes
from Model import Model


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



data = NaturalScenes('D:/projects/data/NaturalScenes')

model = Model(1, module_list)
model.optimizer = optim.Adam(model.parameters(), lr=0.1)
model.loss_fn = nn.NLLLoss()


model.test_model(data.loaders)

training_output = model.run(data.loaders, num_epochs=2)


