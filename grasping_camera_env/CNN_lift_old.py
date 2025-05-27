import torch
import torch.nn as nn

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed()  # e.g. `set_seed(42)` for fixed seed


# define shared model (stochastic and deterministic models) using mixins
class Shared(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)
        print("Observation_space")
        print(observation_space)
        print(self.num_observations)

        self.cnn = nn.Sequential(               
            nn.Conv2d(4, 32, kernel_size=3, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            #nn.Linear(6912, 128),
            #nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
        )
            
        """self.cnn2 = nn.Sequential(               
            nn.Conv2d(4, 32, kernel_size=3, stride=2),
            #nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            #nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2),
            #nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            #nn.Linear(6912, 128),
            #nn.ELU(),
            nn.Linear(128, 64),
            nn.ELU(),
            nn.Linear(64, 32),
            nn.ELU(),
                                                )"""

        self.net = nn.Sequential(nn.Linear(32+18, 256),
                                 nn.ELU(),
                                 nn.Linear(256, 128),
                                 nn.ELU(),
                                 nn.Linear(128, 64),                                
                                 nn.ELU())

        self.mean_layer = nn.Linear(64, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.ones(self.num_actions))

        self.value_layer = nn.Linear(64, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):

        #print(env.observation_space)
        observations = self.tensor_to_space(inputs["states"], self.observation_space)

        #depth_img = observations["depth_img"].permute(0, 3, 1, 2)
        #depth_features = self.cnn(depth_img)

        rgb_img = observations["rgb_img"].permute(0, 3, 1, 2)   # Shape: [B, 3, H, W]
        #rgb_img2 = observations["rgb_img2"].permute(0, 3, 1, 2)   # Shape: [B, 3, H, W]
        depth_img = observations["depth_img"].permute(0, 3, 1, 2) # Shape: [B, 1, H, W]
        #depth_img2 = observations["depth_img2"].permute(0, 3, 1, 2) # Shape: [B, 1, H, W]
        combined_img = torch.cat([rgb_img, depth_img], dim=1)     # Shape: [B, 4, H, W]
        #combined_img2 = torch.cat([rgb_img2, depth_img2], dim=1)     # Shape: [B, 4, H, W]
        #combined_img = torch.cat([depth_img, depth_img2], dim=1)     # Shape: [B, 4, H, W]
        depth_features = self.cnn(combined_img)
        #depth_features2 = self.cnn2(combined_img2)
        #depth_features = self.cnn(depth_img)
        #depth_features_comb = torch.cat([depth_features, depth_features2], dim=1)

        non_image_obs = [v.view(v.size(0), -1) for k, v in observations.items() if k not in ['depth_img', 'rgb_img']]
        non_image_obs = torch.cat(non_image_obs, dim=1)

        #non_image_obs = [v.view(v.size(0), -1) for k, v in observations.items() if k != 'depth_img']
        #non_image_obs = torch.cat(non_image_obs, dim=1)

        #print(non_image_obs)

        combined_features = torch.cat([depth_features, non_image_obs], dim=1)
        #print("combined features size: ", combined_features.shape)

        shared_output = self.net(combined_features)

        if role == "policy":
            #self._shared_output = self.net(inputs["states"])
            return self.mean_layer(shared_output), self.log_std_parameter, {}
        elif role == "value":
            #shared_output = self.net(inputs["states"]) if self._shared_output is None else self._shared_output
            #self._shared_output = None
            return self.value_layer(shared_output), {}


# load and wrap the Isaac Lab environment
env = load_isaaclab_env(task_name="Isaac-Rock-Grasp-v0") #Isaac-Lift-Cube-Franka-v0
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
cfg["learning_epochs"] = 18
cfg["mini_batches"] = 4  # 96 * 4096 / 98304
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 0.001 #0.005
cfg["learning_rate_scheduler"] = KLAdaptiveLR
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01, "min_lr": 1e-5}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.01 #0.01
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
cfg["experiment"]["checkpoint_interval"] = 2000
cfg["experiment"]["directory"] = "p6/runs/Isaac-Lift-Franka-v0"
cfg["experiment"]["wandb"] = True                   #aktivere wandb
cfg["experiment"]["wandb_kwargs"] ={                #Ting der bliver givet til wandb init, meget smart gutter
    "entity": "urkanin-aalborg-universitet",        #Hvilken konto/teams det skal gemmes på, det her er vores fælles
    "project": "P6",                                #Hvilket projekt inde på teams det skal gemmes på
    "group": "Grasp-CNN-test",                     #Man kan gruppere sine runs, smart hvis man tester forksellige ting af, og skal have et samlet overblik over netop dem
    "job_type": "train"                             #Synes vi skal have den her til train/eval, så kan man nemt skelne
} 

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)


# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 45000, "headless": True}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()


# # ---------------------------------------------------------
# # comment the code above: `trainer.train()`, and...
# # uncomment the following lines to evaluate a trained agent
# # ---------------------------------------------------------

#path = "p6/runs/Isaac-Lift-Franka-v0/25-05-21_10-51-30-716918_PPO/checkpoints/best_agent.pt"
#agent.load(path)

# # start evaluation
#trainer.eval()
