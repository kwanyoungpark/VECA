# VECA-python Tutorial

* If you didn't installed VECA-unity yet, please install VECA-unity and finish the VECA-unity tutorial first.*

Here's a minimum example of running VECA-unity application in python side using VECA-python. (IP, PORT indicates IP and port of python side.)

```python
import numpy as np
from VECA import GeneralEnvironment
NUM_AGENTS = 1
ACTION_LENGTH = 2
env = GeneralEnvironment(NUM_AGENTS = NUM_AGENTS, port = PORT)
env.reset()
for _ in range(1000): 
    env.step(2 * np.random.rand(NUM_AGENTS, ACTION_LENGTH) - 1) # Random actions
env.close()
```
Run the python code and wait until prints "0.0.0.0".

```console
$ python test.py
0.0.0.0
```

Then execute VECA-unity application using:

```console
$ ./VECAUnityAppName -ip IP -port PORT 
```

In the python side, it would print message like this:

```console
CONNECTED
b'\x01\x00\x00\x00'
GO 
```

And you will observe the baby agent repeating random actions.

## Observations

 Since you learned how to send actions to the environment, let's learn how to get observations from the environment. On the unity applications, you would coded *CollectObservations()* with bunch of *AddObservations(key, observation)*. Those data are converted into python dictionary object, which *observation* is a value corresponding to the *key*. The *step(self, action)* function included in *GeneralEnvironment* class returns this dictionary object.

  Although this representation is not bad, we can't apply hundreds of codes made for gym environments. To make KickTheBall environment similar to gym environment, we will code the *step(self, action)* function of the wrapper class to return four values: *observation*, *reward*, *done*, *info*.

First, navigate to the Example folder to use example codes.

```
$ cd path/to/rootdirectory/python/Examples
```

We will use *Environment* class included in *KickTheBallEnv.py*.

```python
import numpy as np
from constants import *
from utils import wav2freq
from environment import GeneralEnvironment

class Environment(GeneralEnvironment):
    def __init__(self, NUM_AGENTS, port):
        GeneralEnvironment.__init__(self, NUM_AGENTS, port)

    def step(self, action):
        data = super().step(action)
        rewards, done, info = [], [], []
        imgs, wavs = [], []
        for i in range(NUM_AGENTS):
            img = list(reversed(data['img'][i]))
            wav = data['wav'][i]
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            pos = data['pos'][i]
            img = np.reshape(np.array(img), [2, IMG_H, IMG_W]) / 255.0
            wav = np.reshape(np.array(wav), [2, -1]) / 32768.0 
            wav = wav2freq(wav)
            imgs.append(img)
            wavs.append(wav)
            rewards.append(reward)
            if doneA: done.append(True)
            else: done.append(False)
            info.append(pos)
        imgs, wavs = np.array(imgs), np.array(wavs)
        obs = (imgs, wavs)
        return (obs, rewards, done, info)
```

As you can see, *Environment* class inherits *GeneralEnvironment* class and redefine *step(self, action)* using *super().step(action)*.

 This is just an implementation of how to wrap the VECA-unity environment to classical gym environment. Although it is not exactly same as gym environment, most of the time it works well!


