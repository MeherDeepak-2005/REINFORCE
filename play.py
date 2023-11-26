import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter
from torch import save as torch_save
from policy_network import Agent
from tqdm import tqdm
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--epochs", defualt=20_000, type=int, help="Number of games to play"
    )
    parser.add_argument(
        "--lr", default=0.0005, help="Learning rate for NN Policy Network", type=float
    )
    parser.add_argument("--logdir", default="./plays/LunarLander-v2", type=str)
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument(
        "--chkpt",
        default="./agent/LunarLander-v2.pt",
        help="Save/Load checkpoint file address for model",
        type=str,
    )
    args = parser.parse_args()

    writer = SummaryWriter(log_dir=args.logdir)
    env = gym.make(args.env)

    agent = Agent(env.observation_space.shape[0], env.action_space.n, lr=args.lr)

    progress_bar = tqdm(total=args.epochs, desc="Playing episode")
    for epoch in range(args.epochs):
        done = False
        obs, info = env.reset()

        while not done:
            action = agent.choose_action(obs)
            obs, reward, truncated, terminated, info = env.step(action)
            done = terminated or truncated

            agent.rewards.append(reward)

            writer.add_scalar("Play/reward", reward, epoch)

        mean_rewards = sum(agent.rewards) / len(agent.rewards)

        loss = agent.learn()

        writer.add_scalar("Play/Loss", loss.item(), epoch)
        writer.flush()

        torch_save(agent.policy.state_dict(), args.chkpt)

        progress_bar.update(1)
        progress_bar.set_postfix_str(
            f"loss - {loss.item()} mean_rewards - {mean_rewards} iteration - {epoch}"
        )

    writer.close()
    env.close()
