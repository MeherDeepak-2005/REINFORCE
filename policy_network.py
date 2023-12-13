import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, n_features, n_actions, lr):
        super(PolicyNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
        )
        self.pi = nn.Linear(256, n_actions)

        if lr is not None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, state: torch.Tensor):
        features = self.layers(state)
        # log value for each action
        action_logs = self.pi(features)

        # calculate the probability of logs
        return F.softmax(action_logs, dim=0)


class Agent:
    def __init__(self, observation_space, action_space, lr):
        self.policy = PolicyNetwork(
            n_features=observation_space, n_actions=action_space, lr=lr
        )
        self.rewards = []
        self.action_probs = []
        self.lr = lr

        self.gamma = 0.99

    def choose_action(self, state):
        state = torch.tensor(state).to(self.policy.device)

        log_probs = self.policy(state)
        action_dist = torch.distributions.Categorical(log_probs)
        action = action_dist.sample()

        # no calculations required during testing.
        if self.lr is not None:
            action_probs: int = action_dist.log_prob(action).unsqueeze(0)
            self.action_probs.append(action_probs)

        return action.item()

    def learn(self):
        G = 0
        returns = []

        for reward in reversed(self.rewards):
            G = self.gamma * G + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float, device=self.policy.device)

        action_probs = torch.cat(self.action_probs)

        # loss = G * log pi (at | st)
        # -loss: since gradient ascent is being performed
        loss = -(returns * action_probs).sum()

        self.policy.optimizer.zero_grad()
        loss.backward()
        self.policy.optimizer.step()

        self.action_probs = []
        self.rewards = []

        return loss
