import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size, activation_function):
        super(FeatureExtractor, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._activation_function = activation_function
        self._fc = nn.Linear(input_size, output_size)
        nn.init.xavier_uniform_(self._fc.weight,
                                gain=nn.init.calculate_gain('relu'))

    def forward(self, input):
        return self._activation_function(self._fc(input))