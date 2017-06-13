from __future__ import print_function
from __future__ import absolute_import
from TFNaviTrpo import NaviTRPO

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import stub, run_experiment_lite

from LDNaviEnvLSTMApril12 import LDNaviEnvLSTMApril12
from LDNaviEnvLSTMApril13 import LDNaviEnvLSTMApril13
from NaviGaussian_mlp_baseline import NaviGaussianMLPBaseline
from TFNaviBatch_sampler import BatchSampler
from NaviLinear_feature_baseline import NaviLinearFeatureBaseline
from rllab.baselines.zero_baseline import ZeroBaseline

from sandbox.rocky.tf.policies.gaussian_lstm_policy import GaussianLSTMPolicy
from sandbox.rocky.tf.policies.gaussian_gru_policy import GaussianGRUPolicy
from NaviGaussian_lstm_policy import NaviGaussianLSTMPolicy
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer, FiniteDifferenceHvp
from sandbox.rocky.tf.core.network import MLP


import lasagne.nonlinearities as NL


from TFMlp_baseline import MLPBaseline
import tensorflow as tf
from sandbox.rocky.tf.envs.base import TfEnv

n_iteration = 1000

stub(globals())

env = TfEnv(normalize(LDNaviEnvLSTMApril12()))

featureNW = MLP(
    input_shape=(env.spec.observation_space.flat_dim,),
    output_dim=64,
    hidden_sizes=(256,64),
    hidden_nonlinearity=tf.tanh,
    output_nonlinearity=tf.tanh,
    name="feature_network"
)

policy = NaviGaussianLSTMPolicy(
    name="policy",
    env_spec=env.spec,
    hidden_dim=64,
    output_nonlinearity=tf.tanh,
    state_include_action=False,
    feature_network=featureNW,
    init_std=0.5,
    n_itr=200.,
    min_std=0.1,

)

# baseline = MLPBaseline(env_spec=env.spec,regressor_args=dict(
#                                        [("hidden_sizes", (256, 128, 64))]))
baseline = NaviGaussianMLPBaseline(env_spec=env.spec,
                                   regressor_args=dict(
                                       [("hidden_sizes", (256,64,16)), ("step_size", 0.1),
                                        ("init_std",0.1)]))

# baseline = NaviLinearFeatureBaseline(env_spec=env.spec)
# baseline = ZeroBaseline(env_spec=env.spec)


algo = NaviTRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=50000,
    max_path_length=400,
    n_itr=n_iteration,
    discount=0.995,
    gae_lambda=0.96,
    step_size=0.02,
    optimizer=ConjugateGradientOptimizer(hvp_approach=FiniteDifferenceHvp(base_eps=1e-8),max_backtracks=20,reg_coeff=1e-8),
    sampler_cls=BatchSampler,
    center_adv=False,
    step_scale=1.
)

run_experiment_lite(
    algo.train(),
    # Number of parallel workers for sampling
    n_parallel=4,
    # Only keep the snapshot parameters for the last iteration
    snapshot_mode="last",
    # Specifies the seed for the experiment. If this is not provided, a random seed
    # will be used
    # seed=1,
    # plot=True,
    exp_name="LSTM,April12,stepScale=1,EndRW=+-10k,10*rotRW, reg_coeff=1e-8,"
)
