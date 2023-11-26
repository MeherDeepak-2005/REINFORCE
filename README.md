## REINFORCE - Policy Gradient

- On Policy Learning
- Advantageous for Continuous State/Action spaces

> Model learns by maximising the expectation of returns G ~ $$\mathbb{E_{\pi}\left[ G_{t} \right]}$$ under the policy $$\pi$$.
> The policy $$\pi \left(a | s, \theta \right)$$ states the probability distrubtion of taking action $$a_{t}$$ given the state $$s_{t}$$ at timestep $$t$$ under the parameters of the Neural network $$\theta$$

### Policy Gradient Theorem
$$ \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t, \theta) G_t \right] $$

**Parameters Update**
- $$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta) $$

- This method is a gradient ascent, since pytorch can only perform gradient descent, the loss is calculated as the negative expectation of returns under the policy.

   $$ L\left(\theta\right) = -\mathbb{E}_{\pi_\theta}\left[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t|s_t, \theta) G_t \right] $$

## Train
**Params**
- Env: Choose a compatible env from gym to train the model, default='LunarLander-v2'
- Learning Rate: Experiment with different learning rates to find the optimal one, default=0.0005
- logdir: Visaulise the ongoing training results using tensorboard

**Make checkpoint directories**
```shell
mkdir -p ./agent/LunarLander-v2
```

**Generate samples**
```shell
python play.py --env 'LunarLander-v2' --logdir './plays/LunarLander-v2/lr=0.0005' --epochs 10000 --lr==0.0005 --chkpt './agent/LunarLander-v2/lr=0.0005.pt' && 
python play.py --env 'LunarLander-v2' --logdir './plays/LunarLander-v2/lr=0.001' --epochs 10000 --lr=0.001 --chkpt './agent/LunarLander-v2/lr=0.001' 
```

**Analyze the training results using Tensorboard**
```shell
tensorboard --logdir=./plays/LunarLander-v2/
```

**Results**
```shell
python play_testing.py --env "LunarLander-v2" --chkpt './agent/LunarLander-v2/lr=0.0005'
```

[Watch the video](./videos/LunarLander-v2.mp4)

