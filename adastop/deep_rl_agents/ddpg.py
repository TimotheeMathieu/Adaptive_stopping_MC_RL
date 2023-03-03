# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ddpg/#ddpg_continuous_actionpy
import argparse
import json
import os
import random
import sys
import time
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


N_EVALS = 20


# Parameters
parameters = dict(
    env_id="MountainCarContinuous-v0",
    budget=50_000,
    learning_rate=0.001,
    buffer_size=1_000_000,
    gamma=0.99,
    tau=0.005,
    batch_size=128,
    exploration_noise=0.5,
    learning_starts=100,
    policy_frequency=32,
    noise_clip=0.5,
    n_eval_episodes=3,
    gpu=False, # Say whether you use GPU!
    gym_version=gym.__version__,
    numpy_version=np.__version__,
    python_version=sys.version,
    torch_version=torch.__version__
)
output_dir_name = "results/ddpg/" # experiment folder
os.makedirs(output_dir_name, exist_ok=True)

locals().update(parameters)  # load all the variables defined in parameters dict


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", "-s", type=int, required=True,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", "-e", type=str, default=env_id,
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=budget,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=learning_rate,
        help="the learning rate of the optimizer")
    parser.add_argument("--buffer-size", type=int, default=buffer_size,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=gamma,
        help="the discount factor gamma")
    parser.add_argument("--tau", type=float, default=tau,
        help="target smoothing coefficient (default: 0.005)")
    parser.add_argument("--batch-size", type=int, default=batch_size,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--exploration-noise", type=float, default=exploration_noise,
        help="the scale of exploration noise")
    parser.add_argument("--learning-starts", type=int, default=learning_starts,
        help="timestep to start learning")
    parser.add_argument("--policy-frequency", type=int, default=policy_frequency,
        help="the frequency of training policy (delayed)")
    parser.add_argument("--noise-clip", type=float, default=noise_clip,
        help="noise clip parameter of the Target Policy Smoothing Regularization")
    parser.add_argument('--n-eval-episodes', type=int, default=n_eval_episodes,
        help='the number of episodes to evaluate the agent')
    args = parser.parse_args()

    # Update parameters dict
    args_dict = vars(args)
    for key, value in args_dict.items():
        if key in parameters and parameters[key] != value:
            parameters[key] = value

    # fmt: on
    return args


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        if capture_video:
            if idx == 0:
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mu = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.single_action_space.high - env.single_action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.single_action_space.high + env.single_action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


def evaluate(actor, env, n_eval_episodes=3, noise_scale=None):
    evaluations = np.zeros(n_eval_episodes)
    for i in range(n_eval_episodes):
        obs, done = env.reset(), False
        while not done:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                if noise_scale is not None:
                    actions += torch.normal(0, actor.action_scale * noise_scale)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)
            obs, reward, done, _ = env.step(actions)
            evaluations[i] += reward
    return evaluations


if __name__ == "__main__":
    args = parse_args()
    locals().update(parameters)

    # create directory for this run
    output_dir_name = os.path.join(output_dir_name, str(args.env_id), str(args.seed))
    os.makedirs(output_dir_name, exist_ok=True)

    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(output_dir_name)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    eval_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed, 0, args.capture_video, run_name)])
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    actor = Actor(envs).to(device)
    qf1 = QNetwork(envs).to(device)
    qf1_target = QNetwork(envs).to(device)
    target_actor = Actor(envs).to(device)
    target_actor.load_state_dict(actor.state_dict())
    qf1_target.load_state_dict(qf1.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()), lr=args.learning_rate)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.learning_rate)

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        handle_timeout_termination=True,
    )
    start_time = time.time()

    # evaluation arrays
    timesteps = np.zeros(N_EVALS + 1, dtype=int)
    evaluations = np.zeros(N_EVALS + 1, dtype=float)
    evaluations[0] = np.mean(evaluate(actor, eval_envs, args.n_eval_episodes, noise_scale=args.exploration_noise))

    # TRY NOT TO MODIFY: start the game
    n_evals, eval_freq = 0, args.total_timesteps // N_EVALS
    obs = envs.reset()
    for global_step in range(args.total_timesteps):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            with torch.no_grad():
                actions = actor(torch.Tensor(obs).to(device))
                actions += torch.normal(0, actor.action_scale * args.exploration_noise)
                actions = actions.cpu().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        for info in infos:
            if "episode" in info.keys():
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `terminal_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(dones):
            if d:
                real_next_obs[idx] = infos[idx]["terminal_observation"]
        rb.add(obs, real_next_obs, actions, rewards, dones, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions = target_actor(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (qf1_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)

            # optimize the model
            q_optimizer.zero_grad()
            qf1_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:
                actor_loss = -qf1(data.observations, actor(data.observations)).mean()
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                # update the target network
                for param, target_param in zip(actor.parameters(), target_actor.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # ALGO LOGIC: evaluation
        if global_step > 0 and global_step % eval_freq == 0:
            evaluation = evaluate(actor, eval_envs, args.n_eval_episodes, noise_scale=args.exploration_noise)

            print(f"global_step={global_step}, evaluation={np.mean(evaluation):.3f} +- {np.std(evaluation):.3f}")
            writer.add_scalar("charts/evaluation", np.mean(evaluation), global_step)

            n_evals += 1
            timesteps[n_evals] = global_step
            evaluations[n_evals] = np.mean(evaluation)
            np.savez(os.path.join(output_dir_name, f"evaluations.npz"), timesteps=timesteps, evaluations=evaluations)

    # save the model
    torch.save(actor.state_dict(), os.path.join(output_dir_name, f"{run_name}_actor.pth"))
    torch.save(qf1.state_dict(), os.path.join(output_dir_name, f"{run_name}_qf1.pth"))
    torch.save(qf1_target.state_dict(), os.path.join(output_dir_name, f"{run_name}_qf1_target.pth"))
    torch.save(target_actor.state_dict(), os.path.join(output_dir_name, f"{run_name}_target_actor.pth"))
    
    # save parameters
    parameters["save_path"] = output_dir_name
    with open(os.path.join(output_dir_name, "parameters.json"), "w") as file:
        file.write(json.dumps(parameters))

    # evaluate the agent last time
    r_mean = np.mean(evaluate(actor, eval_envs, args.n_eval_episodes, noise_scale=args.exploration_noise))
    print(f"AdaStop Evaluation: {r_mean}")

    envs.close()
    writer.close()
