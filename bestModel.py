
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim
import torch
import csv

from naturalScenesData import NaturalScenesPredications
from Model import Model


module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.Sigmoid(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    )
])

id = 1
outputFile = "bestModel"
optimizer_class = optim.NAdam
loss_fn_class = nn.NLLLoss


model = Model(id, outputFile, module_list, optimizer_class, loss_fn_class)
model.optimizer = optimizer_class(model.parameters(), lr=0.0005)

model.load_state_dict(torch.load('testResults/testsFour/BestModel/state_dics/model_1_run_10_uuid_9802706af94811eda9b528d0ea2e314f'))
model.eval()

data = NaturalScenesPredications('D:/programming/data/NaturalScenes/seg_pred')

with open('predictions.csv', 'w', newline='') as file:
    writer = csv.writer(file)

    with torch.no_grad():
        for i, (name, d) in enumerate(data.images):

            output = model(d)
            writer.writerow([name, output.argmax(dim=1).item()])

