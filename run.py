
# created by Ryan Davis, c3414318, ryan_davis00@hotmail.com

from torch import nn, optim

from naturalScenesData import NaturalScenes, NaturalScenesSubset
from Model import Model
import ModuleLists



using2d = ModuleLists.Using2d()


data = NaturalScenesSubset('D:/projects/data/NaturalScenes/seg_train', 'D:/projects/data/NaturalScenes/seg_test')

model = Model(2, 'using2d', using2d.module_list_2, optim.Adam, nn.NLLLoss)
model.run(*data.loaders, title="2d logsoftmax")
model = Model(1, 'using2d', using2d.module_list_1, optim.Adam, nn.NLLLoss)
model.run(*data.loaders, title="1d logsoftmax")