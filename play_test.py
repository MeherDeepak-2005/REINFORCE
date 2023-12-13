import gymnasium as gym
from policy_network import Agent
from torch import load as torch_load
from argparse import ArgumentParser
import os

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--env", default="LunarLander-v2", type=str)
    parser.add_argument(
        "--chkpt",
        default="./agent/",
        help="Save/Load checkpoint file address for model",
        type=str,
    )
    parser.add_argument(
        "--render_mode", default="human", help="Render Mode for Env", type=str
    )
    args = parser.parse_args()
    chkpt = os.path.join(args.chkpt, f"{args.env}.pt")

    env = gym.make(args.env, render_mode=args.render_mode)
    agent = Agent(env.observation_space.shape[0], env.action_space.n, lr=None)
    agent.policy.load_state_dict(torch_load(chkpt))
    agent.policy.eval()

    while True:
        play_one = input("Play game - [y/N] ")
        if play_one == "y":
            done = False
            obs, info = env.reset()

            while not done:
                action = agent.choose_action(obs)
                obs, reward, truncated, terminated, info = env.step(action)

                done = truncated or terminated
        else:
            break
