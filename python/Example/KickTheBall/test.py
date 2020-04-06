import numpy as np
from VECA.environment import GeneralEnvironment
NUM_AGENTS, PORT = 1, 8870
ACTION_LENGTH = 2
env = GeneralEnvironment(NUM_AGENTS = NUM_AGENTS, port = PORT)
env.reset()
for _ in range(1000): 
	env.step(2 * np.random.rand(NUM_AGENTS, ACTION_LENGTH) - 1) # Random actions
env.close()

