from gym.envs.registration import register

register(
    id='ConsensusEnv-v0',
    entry_point='ConsensusEnv.envs:ConsensusEnv',
)

register(
    id='ConsensusContEnv-v0',
    entry_point='ConsensusEnv.envs:ConsensusContEnv',
)

register(
    id='CentralizedConsensusContEnv-v0',
    entry_point='ConsensusEnv.envs:CentralizedConsensusContEnv',
)

# register(
#     id='ConsensusEnv-extrahard-v0',
#     entry_point='ConsensusEnv.envs:ConsensusEnvExtraHardEnv',
# )