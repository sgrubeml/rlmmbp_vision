import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from learned_robot_placement.models.feature_extractor import FeatureExtractor


class CriticNetwork_RNN(nn.Module):
    def __init__(self, input_shape, output_shape, n_features, **kwargs):
        super().__init__()

        self._obs_feature_extractor = kwargs['embedding']
        self._n_channels = kwargs['channels']
        self._obs_dim = kwargs['obs_dim']

        self._internal_state = None

        self._gru_hidden_size = kwargs['gru_hidden_size']
        gru_input_size = kwargs['gru_input_size']
        gru_layers = kwargs['gru_layers']

        self._device = 'cuda:0'

        self._robot_state_extractors = nn.ModuleList()
        robot_state_embeddings = kwargs['robot_state_embeddings']

        n_input = input_shape[-1]
        n_output = output_shape[0]

        for embedding in robot_state_embeddings:
            if embedding['collect']:
                input_size, output_size = embedding["input"], embedding["output"]
                self._robot_state_extractors.append(FeatureExtractor(input_size, output_size, F.relu))
        
        self._gru = nn.GRU(input_size=gru_input_size,
                            hidden_size=self._gru_hidden_size,
                            num_layers=gru_layers,
                            batch_first=True,
                            bias=True,
                            dtype=torch.float)
        
        self._h1 = nn.Linear(n_input, n_features)
        self._h2 = nn.Linear(n_features, n_features)
        self._h3 = nn.Linear(n_features, n_output)

        nn.init.xavier_uniform_(self._h1.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h2.weight,
                                gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self._h3.weight,
                                gain=nn.init.calculate_gain('linear'))

    def forward(self, state, action):
        obs = state[:,:self._obs_dim].view(-1, self._obs_dim//self._n_channels, self._n_channels)
        obs = torch.permute(obs, (0,2,1))
        # get observation embedding
        embedding = self._obs_feature_extractor(obs)
        # get rl state feature embeddings
        state_index = self._obs_dim
        for feature_extractor in self._robot_state_extractors:
            embedding = torch.cat((embedding,feature_extractor(state[:,state_index:state_index+feature_extractor._input_size])), 1)
            state_index += feature_extractor._input_size
        
        gru_output, _ = self._gru(embedding)
        features1 = F.relu(self._h1(torch.hstack((gru_output, action.float()))))
        features2 = F.relu(self._h2(features1))
        q = self._h3(features2)
        return torch.squeeze(q)
