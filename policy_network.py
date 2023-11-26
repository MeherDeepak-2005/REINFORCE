import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, n_features, n_actions, lr):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(n_features, 128)
        self.fc2 = nn.Linear(128, 128)

        self.pi = nn.Linear(128, n_actions)

        if lr is not None:
            self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.pi(x)


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

        action_logs = F.softmax(self.policy(state), dim=0)
        action_dist = torch.distributions.Categorical(action_logs)
        action = action_dist.sample()

        if self.lr is not None:
            action_probs = action_dist.log_prob(action).unsqueeze(0)
            self.action_probs.append(action_probs)

        return action.item()

    def learn(self):
        self.policy.optimizer.zero_grad()
        G = 0
        returns = []

        # backward accumulation of returns G
        # faster than traditional approach
        for reward in reversed(self.rewards):
            G = self.gamma * G + reward
            returns.append(G)

        returns.reverse()
        returns = torch.tensor(returns, dtype=torch.float, device=self.policy.device)

        action_probs = torch.cat(self.action_probs)

        # loss = - G * log pi (at | st)
        loss = -(returns * action_probs).sum()
        loss.backward()
        self.policy.optimizer.step()

        self.action_probs = []
        self.rewards = []

        return loss
