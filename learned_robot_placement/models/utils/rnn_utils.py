from mushroom_rl.utils.callbacks.callback import Callback


class RNNinternalStateCallbackStep(Callback):
    """
    This callback can be used to set the internal state of the RNN 

    """
    def __init__(self, networks: list):
        self._networks = networks
    def __call__(self, sample):
        for network in self._networks:
            network.model.network.reset_internal_state(sample[0][-1])

class RNNinternalStateCallbackFit(Callback):
    """
    This callback can be used to set the internal state of the RNN 

    """
    def __init__(self, networks: list):
        self._networks = networks
    def __call__(self, dataset):
        for network in self._networks:
            network.model.network.reset_internal_state(True)
