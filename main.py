# import argparse

import numpy as np

# custom modules
from utils.options import Options
from utils.factory import EnvDict, ModelDict, MemoryDict, AgentDict


# def parse_args():
#     parser = argparse.ArgumentParser()
#
#     parser.add_argument(
#         '--machine',
#         type=str,
#         default=None,
#         help='TODO'
#     )
#     parser.add_argument(
#         '--timestamp',
#         type=str,
#         default=None,
#         help='TODO'
#     )
#     parser.add_argument(
#         '--step',
#         type=str,
#         default=None,
#         help='TODO'
#     )
#     return parser.parse_args()
#
# args = parse_args()
# if args.machine is not None and args.timestamp is not None and args.step is not None:
#     opt.

# 0. setting up
opt = Options()
np.random.seed(opt.seed)

# 1. env    (prototype)
env_prototype = EnvDict[opt.env_type]
# 2. model  (prototype)
model_prototype = ModelDict[opt.model_type]
# 3. memory (prototype)
memory_prototype = MemoryDict[opt.memory_type]
# 4. agent
agent = AgentDict[opt.agent_type](opt.agent_params,
                                  env_prototype=env_prototype,
                                  model_prototype=model_prototype,
                                  memory_prototype=memory_prototype)
# 5. fit model
if opt.mode == 1:  # train
    agent.fit_model()
elif opt.mode == 2:  # test opt.model_file
    agent.test_model()
