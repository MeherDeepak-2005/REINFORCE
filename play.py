import numpy as np
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch import save as torch_save
from policy_network import Agent
from tqdm import tqdm
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", default=20_000, type=int, help="Number of games to play"
    )
    parser.add_argument(
        "--lr", default=0.0005, help="Learning rate for NN Policy Network", type=float
    )
    parser.add_argument("--logdir", default="./plays", type=str)
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument(
        "--chkpt",
        default="./agent",
        help="Save/Load checkpoint file address for model",
        type=str,
    )
    args = parser.parse_args()

    log_dir = os.path.join(args.logdir, args.env)
    chkpt = os.path.join(args.chkpt, f"{args.env}.pt")

    writer = SummaryWriter(log_dir=log_dir)
    env = gym.make(args.env)

    agent = Agent(env.observation_space.shape[0], env.action_space.n, lr=args.lr)

    print("RunTime Details: ")
    print(f"   > Playing - {args.env} for {args.epochs} episodes")
    print(f"   > TensorBoard Logdir - {log_dir} Checkpoint File - {chkpt}")

    progress_bar = tqdm(total=args.epochs, desc="Playing episode")
    episode_rewards = np.zeros((args.epochs, 1))
    episode_losses = np.zeros((args.epochs, 1))
    for epoch in range(args.epochs):
        done = False
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs, reward, truncated, terminated, info = env.step(action)
            done = terminated or truncated

            agent.rewards.append(reward)

        episode_rewards[epoch] = sum(agent.rewards)
        loss = agent.learn()
        episode_losses[epoch] = loss.cpu().detach().numpy()

        if epoch > 100:
            # log mean of losses, rewards of last 100 episodes
            writer.add_scalar(
                "Play/mean_loss", episode_losses[epoch - 100 : epoch].mean(), epoch
            )
            writer.add_scalar(
                "Play/mean_rewards", episode_rewards[epoch - 100 : epoch].mean(), epoch
            )
            writer.add_scalar("Play/episode_rewards", episode_rewards[epoch], epoch)
            writer.flush()

            progress_bar.set_postfix_str(
                f"episode_reward - {float(episode_rewards[epoch]):.2f} "
                f"mean_rewards - {float(episode_rewards[epoch-100: epoch].mean()):.2f} "
                f"mean_loss - {float(episode_losses[epoch-100: epoch].mean()):.2f}"
            )

        torch_save(agent.policy.state_dict(), chkpt)

        progress_bar.update(1)

    writer.close()
    env.close()
