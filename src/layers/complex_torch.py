import torch

class ComplexDense(torch.nn.Module):
    def __init__(self, \
        in_features,
        out_features,
        activation = 'elu',
        ):
        super(ComplexDense, self).__init__()
        self.fc = torch.nn.Linear(
            in_features,
            out_features
        )
