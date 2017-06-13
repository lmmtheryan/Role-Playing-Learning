import numpy as np
from rllab.misc import tensor_utils


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    # Feb 3, 2017
    states = []
    # o = env.reset()
    o, s = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)

        next_o, r, d, env_info = env.step(a)

        observations.append(env.observation_space.flatten(o))

        states.append(env.observation_space.flatten(s))

        rewards.append(r)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o

        s = env_info["states"]

        if animated:
            env.render()
    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        states = tensor_utils.stack_tensor_list(states),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
