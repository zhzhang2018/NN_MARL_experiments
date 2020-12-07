U_VELOCITY = 1
U_ACCELERATION = 2
O_VELOCITY = 1
O_ACCELERATION = 2

O_ACTION = 1
O_NO_ACTION = 0

DIST_REWARD = 0x1
TIME_REWARD = 0x2
ACT_REWARD = 0x4
ALL_REWARD = DIST_REWARD | TIME_REWARD | ACT_REWARD

# Boundary policies
NO_PENALTY = 0
SOFT_PENALTY = 1 # Very light penalty
HARD_PENALTY = 2 # Very expensive penalty
DEAD_ON_TOUCH = 3 # Terminates simulation when one of them is out of bound, together with hard penalty

# Reward for achieving consensus policies
END_ON_CONSENSUS = 0
REWARD_IF_CONSENSUS = 1 # Doesn't stop, but keeps giving positive reward when seen as achieved consensus
