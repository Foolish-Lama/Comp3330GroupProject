from torch import nn, optim

class ActivationFunctions:

    loss_fn_class = nn.NLLLoss
    optimizer_class = optim.Adam

    module_list_1_title = "RELU"
    module_list_1 = nn.ModuleList([
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

    module_list_2_title = "Sigmoid"
    module_list_2 = nn.ModuleList([
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
        ),
    ])

    module_list_3_title = "Tanh"
    module_list_3 = nn.ModuleList([
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

    module_list_4_title = "LeakyReLU"
    module_list_4 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.Dropout(),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_5_title = "ELU"
    module_list_5 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.Dropout(),
            nn.LogSoftmax(dim=1)
        ),
    ])

    all_module_lists = [
        module_list_1,
        module_list_2,
        module_list_3,
        module_list_3,
        module_list_4,
        module_list_5,
    ]

    all_module_titles = [
        module_list_1_title,
        module_list_2_title,
        module_list_3_title,
        module_list_4_title,
        module_list_5_title,
    ]

class LossFunctions():

    optimizer_class = optim.Adam

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

    all_loss_functions = [
        nn.L1Loss,
        nn.MSELoss,
        nn.NLLLoss,
        nn.CrossEntropyLoss,
        nn.HingeEmbeddingLoss,
        nn.MarginRankingLoss,
        nn.TripletMarginLoss,
        nn.KLDivLoss
    ]

    all_loss_function_titles = [
        "L1Loss",
        "MSELoss",
        "NLLLoss",
        "CrossEntropyLoss",
        "HingeEmbeddingLoss",
        "MarginRankingLoss",
        "TripletMarginLoss",
        "KLDivLoss"
    ]

class Optimizers():

    loss_fn_class = nn.NLLLoss

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

    all_optimzers = [
        optim.Adadelta,
        optim.Adagrad,
        optim.Adam,
        optim.AdamW,
        optim.SparseAdam,
        optim.Adamax,
        optim.ASGD,
        optim.LBFGS,
        optim.NAdam,
        optim.RAdam,
        optim.RMSprop,
        optim.Rprop,
        optim.SGD
    ]
    all_optimzers_titles = [
        "Adadelta",
        "Adagrad",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adamax",
        "ASGD",
        "LBFGS",
        "NAdam",
        "RAdam",
        "RMSprop",
        "Rprop",
        "SGD"
    ]

class Using2d():

    loss_fn_class = nn.NLLLoss
    optimizer_class = optim.Adam

    module_list_1_title = "1d logsoftmax"
    module_list_1 = nn.ModuleList([
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
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_2_title = "a 3d logsoftmax"
    module_list_2 = nn.ModuleList([
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
            nn.LogSoftmax(dim=3),
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_3_title = "1d dropout"
    module_list_3 = nn.ModuleList([
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

    module_list_4_title = "3d dropout"
    module_list_4 = nn.ModuleList([
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
            nn.Dropout3d(),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_5_title = "many 3d dropout"
    module_list_5 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.Dropout3d(),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.Dropout3d(),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.Dropout3d(),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.LogSoftmax(dim=3)
        ),
    ])

    all_module_lists = [
        module_list_1,
        module_list_2,
        module_list_3,
        module_list_4,
        module_list_5,
    ]

    all_module_titles = [
        module_list_1_title,
        module_list_2_title,
        module_list_3_title,
        module_list_4_title,
        module_list_5_title
    ]

class NumLayersOne():
    loss_fn_class = nn.NLLLoss
    optimizer_class = optim.Adam

    module_list_1_title = "7 layers"
    module_list_1 = nn.ModuleList([
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
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, "same"),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(3)
        ),
        nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, "same"),
            nn.BatchNorm2d(512),
            nn.Tanh(),
            nn.MaxPool2d(3)
        ),
        nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, "same"),
            nn.BatchNorm2d(1024),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_2_title = "5 layers"
    module_list_2 = nn.ModuleList([
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
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(2)
        ),
        nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, "same"),
            nn.BatchNorm2d(256),
            nn.Tanh(),
            nn.MaxPool2d(9)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_3_title = "4 layers"
    module_list_3 = nn.ModuleList([
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
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, "same"),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(9)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_4_title = "3 layers"
    module_list_4 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, "same"),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(9)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_5_title = "2 Layers"
    module_list_5 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(16)
        ),
        nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, "same"),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.MaxPool2d(9)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(32, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_6_title = "1 layer"
    module_list_6 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(144)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    all_module_lists = [
        module_list_1,
        module_list_2,
        module_list_3,
        module_list_4,
        module_list_5,
        module_list_6,
    ]
    all_module_list_titles = [
        module_list_1_title,
        module_list_2_title,
        module_list_3_title,
        module_list_4_title,
        module_list_5_title,
        module_list_6_title,
    ]


class NumLayersTwo():
    loss_fn_class = nn.NLLLoss
    optimizer_class = optim.Adam

    module_list_1_title = "20 hidden layers"
    module_list_1 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_2_title = "10 hidden layers"
    module_list_2 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_3_title = "3 hidden layers"
    module_list_3 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_4_title = "1 hidden layers"
    module_list_4 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Conv2d(16, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    module_list_5_title = "0 hidden layers"
    module_list_5 = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, "same"),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.MaxPool2d(4)
        ),
        nn.Sequential(
            nn.Flatten(),
            nn.Linear(20736, 6),
            nn.LogSoftmax(dim=1)
        ),
    ])

    all_module_lists = [
        module_list_1,
        module_list_2,
        module_list_3,
        module_list_4,
        module_list_5
    ]
    all_module_list_titles = [
        module_list_1_title,
        module_list_2_title,
        module_list_3_title,
        module_list_4_title,
        module_list_5_title,
    ]
