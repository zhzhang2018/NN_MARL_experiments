from gym.envs.registration import register

register(
    id='ConsensusEnv-v0',
    entry_point='ConsensusEnv.envs:ConsensusEnv',
)
# register(
#     id='ConsensusEnv-extrahard-v0',
#     entry_point='ConsensusEnv.envs:ConsensusEnvExtraHardEnv',
# )