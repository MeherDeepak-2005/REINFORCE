## REINFORCE - Policy Gradient

- On Policy Learning
- Advantageous for Continuous State/Action spaces
- 
![image](https://github.com/MeherDeepak-2005/REINFORCE/assets/68017128/841947c6-6770-4074-92ab-7d2bce33030a)

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

https://github.com/MeherDeepak-2005/REINFORCE/assets/68017128/b19678d5-5815-43de-9c71-2fb624cd64c6
