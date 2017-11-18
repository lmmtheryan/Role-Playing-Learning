import numpy as np

from rllab.core.serializable import Serializable
from rllab.core.parameterized import Parameterized
from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
from rllab.misc import logger
from rllab.regressors.gaussian_mlp_regressor import GaussianMLPRegressor


class NaviGaussianMLPBaseline(Baseline, Parameterized, Serializable):

    def __init__(
            self,
            env_spec,
            regressor_args=None,

    ):
        Serializable.quick_init(self, locals())
        super(NaviGaussianMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()
        print env_spec.observation_space.flat_dim
        self._regressor = GaussianMLPRegressor(
            input_shape=(6*env_spec.observation_space.flat_dim,),
            output_dim=1,
            name="vf",
            **regressor_args
        )

    @overrides
    def fit(self, paths):
        states = np.concatenate([p["states"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(states, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        return self._regressor.predict(path["states"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        self._regressor.set_param_values(flattened_params, **tags)
