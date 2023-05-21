from torch import nn, optim

class ActivationFunctions:

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
        "RELU",
        "Sigmoid",
        "Tanh",
        "LeakyReLU",
        "ELU",
    ]

class LossFunctions():

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


class Using2d():

    loss_fn_class = nn.NLLLoss
    optimizer_class = optim.Adam

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
        "1d logsoftmax",
        "3d logsoftmax",
        "1d dropout",
        "3d dropout",
        "many 3d dropout"
    ]
