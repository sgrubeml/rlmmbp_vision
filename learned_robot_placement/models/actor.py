import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from learned_robot_placement.models.feature_extractor import FeatureExtractor


class ActorNetwork(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super(ActorNetwork, self).__init__()

        self._state_embedding = kwargs['embedding']
        self._map_size = kwargs['map_size']
        self._n_channels = kwargs['channels']
        self._rl_state_shape = kwargs['rl_state_shape']

        n_input = input_shape[-1]
        n_output = output_shape[0]

        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state):
        if state.shape[0] == 1: self._state_embedding.eval()
        else: self._state_embedding.train()
        pcl, rl_input = torch.split(state, [state.shape[-1] - self._rl_state_shape, self._rl_state_shape], 1)

        pcl = pcl.view(-1, pcl.shape[-1]//self._n_channels, self._n_channels)
        pcl = torch.permute(pcl, (0,2,1))

        encoder_features = self._state_embedding(pcl)
        actor_input = torch.hstack((encoder_features, rl_input))
        features1 = F.relu(self._h1(torch.squeeze(actor_input, 1).float()))
        features2 = F.relu(self._h2(features1))
        a = self._h3(features2)
        return a

