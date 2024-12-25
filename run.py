from torch import nn


class DNN(nn.Module):
    def __init__(self, input_dim):
        super(DNN, self).__init__()

        # TODO: Define the better model architecture
        l1 = nn.Linear(input_dim, 64)
        relu = nn.ReLU()
        l2 = nn.Linear(64, 1) # shape: (batch_size, 1)

        net = nn.Sequential(
            l1,
            relu,
            l2
        )
        # criterion: loss function
        self.criteria = nn.MSELoss(reduction='mean')

    def forward(self, x):
        return self.net(x).Squeeze(1) # shape: (batch_size, 1) -> (batch_size,)

