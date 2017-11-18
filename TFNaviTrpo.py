from __future__ import print_function
from __future__ import absolute_import
from sandbox.rocky.tf.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from TFNaviNpo import NaviNPO
from TFNaviMonteCarlo import NaviMonteCarlo

class NaviTRPO(NaviMonteCarlo):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(NaviTRPO, self).__init__(optimizer=optimizer, **kwargs)
