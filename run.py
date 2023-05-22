
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists



failed_tests = []

data = NaturalScenesSubset('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

LossFunctions = ModuleLists.LossFunctions()

for lf, title, i in zip(LossFunctions.all_loss_functions, LossFunctions.all_loss_function_titles, range(1, len(LossFunctions.all_loss_functions))):
    try:
        model = Model(i, 'lossFunctions', LossFunctions.module_list, LossFunctions.optimizer_class, lf)
        model.run(*data.loaders, title=title)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")


Using2d = ModuleLists.Using2d()

for ml, title, i in zip(Using2d.all_module_lists, Using2d.all_module_titles, range(1, len(Using2d.all_module_lists))):
    try:
        model = Model(i, 'twoDimensionalLayers', ml, Using2d.optimizer_class, Using2d.loss_fn_class)
        model.run(*data.loaders, title=title)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")


NumLayersOne = ModuleLists.NumLayersOne()

for ml, title, i in zip(NumLayersOne.all_module_lists, NumLayersOne.all_module_list_titles, range(1, len(NumLayersOne.all_module_lists))):
    try:
        model = Model(i, 'NumLayersOne', ml, NumLayersOne.optimizer_class, NumLayersOne.loss_fn_class)
        model.run(*data.loaders, title=title)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")


NumLayersTwo = ModuleLists.NumLayersTwo()

for ml, title, i in zip(NumLayersTwo.all_module_lists, NumLayersTwo.all_module_list_titles, range(1, len(NumLayersTwo.all_module_lists))):
    try:
        model = Model(i, 'NumLayersTwo', ml, NumLayersTwo.optimizer_class, NumLayersTwo.loss_fn_class)
        model.run(*data.loaders, title=title)
    except:
        s = str(i) + " " + title
        failed_tests.append(s)
        print(s + "failed")



module_list = nn.ModuleList([
    nn.Sequential(
        nn.Conv2d(3, 16, 3, 1, "same"),
        nn.BatchNorm2d(16),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(16, 32, 3, 1, "same"),
        nn.BatchNorm2d(32),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Conv2d(32, 64, 3, 1, "same"),
        nn.BatchNorm2d(64),
        nn.Tanh(),
        nn.MaxPool2d(2)
    ),
    nn.Sequential(
        nn.Flatten(),
        nn.Linear(20736, 6),
        nn.Dropout(),
        nn.LogSoftmax(dim=1)
    ),
])
outputFile = "learningRate"
optimizer_class = optim.Adam
loss_fn_class = nn.NLLLoss

title = "learningRate 100"
i = 1
try:
    model = Model(i, outputFile, module_list, optimizer_class, loss_fn_class)
    model.optimizer = optim.Adam(model.parameters(), lr=100)
    model.run(*data.loaders, title=title)
except:
    s = str(i) + " " + title
    failed_tests.append(s)
    print(s + "failed")

title = "learningRate 1"
i = 2
try:
    model = Model(i, outputFile, module_list, optimizer_class, loss_fn_class)
    model.optimizer = optim.Adam(model.parameters(), lr=1)
    model.run(*data.loaders, title=title)
except:
    s = str(i) + " " + title
    failed_tests.append(s)
    print(s + "failed")

title = "learningRate 0.1"
i = 3
try:
    model = Model(i, outputFile, module_list, optimizer_class, loss_fn_class)
    model.optimizer = optim.Adam(model.parameters(), lr=0.1)
    model.run(*data.loaders, title=title)
except:
    s = str(i) + " " + title
    failed_tests.append(s)
    print(s + "failed")

title = "learningRate 0.001"
i = 4
try:
    model = Model(i, outputFile, module_list, optimizer_class, loss_fn_class)
    model.optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.run(*data.loaders, title=title)
except:
    s = str(i) + " " + title
    failed_tests.append(s)
    print(s + "failed")

data = NaturalScenes('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')
