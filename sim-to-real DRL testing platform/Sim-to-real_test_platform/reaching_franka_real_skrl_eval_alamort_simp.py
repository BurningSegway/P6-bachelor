import torch
import torch.nn as nn

# Import the skrl components to build the RL system
#from skrl.models.torch import Model, GaussianMixin

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed
from frankx import Robot, Gripper



# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed

# Define only the policy for evaluation
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        print("Observation_space")
        print(observation_space)
        print(self.num_observations)

        self.net = nn.Sequential(nn.Linear(self.num_observations, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(64, 1)


    def act(self, inputs, role):
        #print("Inputs")
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        if role == "policy":
            self._shared_output = self.net(inputs["states"])
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":# instantiate a memory as rollout buffer (any memory can be used for this)
            memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)
            shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}


# Load the environment (env file)
from reaching_franka_real_env_alamort_simp.py import ReachingFranka
#from reaching_franka_real_env_alamort_simp_trying_out_moves import ReachingFranka # from old code i think?
control_space = "joint"   # cartesian or joint
motion_type = "waypoint"  # waypoint or impedance
camera_tracking = False   # True for USB-camera tracking

env = ReachingFranka(robot_ip="192.168.2.30",
                     device="cpu",
                     control_space=control_space,
                     motion_type=motion_type,
                     camera_tracking=camera_tracking)

# wrap the environment
env = wrap_env(env)

device = env.device

# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=96, num_envs=env.num_envs, device=device)

# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
models["policy"] = Shared(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 96  # memory_size
cfg["learning_epochs"] = 10
cfg["mini_batches"] = 4  # 96 * 4# instantiate a memory as rollout buffer (any memory can be used for this)
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 0.005
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = True
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 336
cfg["experiment"]["checkpoint_interval"] = 1000
cfg["experiment"]["directory"] = "p6/runs/Isaac-Lift-Franka-v0"
cfg["experiment"]["wandb"] = False                   #aktivere wandb
cfg["experiment"]["wandb_kwargs"] ={                #Ting der bliver givet til wandb init, meget smart gutter
    "entity": "urkanin-aalborg-universitet",        #Hvilken konto/teams det skal gemmes på, det her er vores fælles
    "project": "P6",                                #Hvilket projekt inde på teams det skal gemmes på
    "group": "Real_life_franka",                     #Man kan gruppere sine runs, smart hvis man tester forksellige ting af, og skal have et samlet overblik over netop dem
    "job_type": "Real_life_tests_with_franka"                             #Synes vi skal have den her til train/eval, så kan man nemt skelne
} 

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# load checkpoints

if control_space == "joint":
    #agent.load("morten_tester_frank/best_agent_simple.pt")
    agent.load("sim-to-real DRL testing platform\Sim-to-real_test_platform\reaching_franka_real_env_alamort_simp.py")
else:
    print("wrong controll space")
# Configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 1000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start evaluation
trainer.eval()
