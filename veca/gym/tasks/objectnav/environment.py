import numpy as np
from veca.gym.core import EnvModule

USE_MENTOR_AUDIO = False
OBJ_NAME = sorted(["toybear", "spoon", "fork", "ball", "cap", "boat", "bus", "cup", "boots", "glove"])
COLOR_NAME = ["Red", "Green", "Blue"]


def wav2freq(wav):
    wav0, wav1 = wav[0], wav[1]
    wav0 = abs(np.fft.rfft(wav0))[:250]
    wav0 = np.log10(wav0 + 1e-8)
    wav1 = abs(np.fft.rfft(wav1))[:250]
    wav1 = np.log10(wav1 + 1e-8)
    wav = np.array([wav0, wav1])
    #print(np.max(wav0), np.max(wav1))
    return wav

class Environment(EnvModule):
    def __init__(self, task, num_envs,  args,
            remote_env, ip, port
        ):
        EnvModule.__init__(self,task, num_envs, args, 
            remote_env, ip, port,
            exec_path_win = "veca\\env_manager\\bin\\objectnav\\VECAUnityApp.exe",
            download_link_win = "https://drive.google.com/uc?export=download&id=15sMnrX4WLib4EZv_1QJAeQorERc6853j",
            exec_path_linux = "./veca/env_manager/bin/objectnav/objectnav.x86_64",
            download_link_linux = "https://drive.google.com/uc?export=download&id=1dhUCy8xDJBwQvZfaU0j_F1LWOaIjvQa0" 
            )
        self.name = 'ObjectNav'
        self.SIM = 'VECA'
        self.mode = 'CONT'
        self.observation_space = {
            'image': (6, 84, 84),
            'obj': 10
        }
        self.record = ("-record" in args)
        self.use_audio = USE_MENTOR_AUDIO
        if self.use_audio:
            self.observation_space['audio'] = (2, 250)
        self.VEC_OBJ = True

    def collect_observations(self, ignore_agent_dim = True):
        data = super().collect_observations(ignore_agent_dim = True)
        rewards, done, info, objs = [], [], [], []
        posAL, posAF, posA, posL, posF = [], [], [], [], []
        imgs = []
        if self.record:
            imgsT = []

        IMG_C, IMG_H, IMG_W = self.observation_space['image']
        if self.use_audio:
            WAV_C, _ = self.observation_space['audio']
            wavs = []
        #print("DATA")
        #print("OBJDATA")
        #print(self.obj_data.keys())
        for i in range(self.num_envs):
            if self.record:
                imgT = list(reversed(data['Recimg'][i]))
                imgT = np.reshape(np.array(imgT), [3, 224, 224]).astype(np.uint8)
                imgsT.append(imgT)
            if 'img' in data:
                img = list(reversed(data['img'][i]))
            else:
                img = np.zeros([IMG_C, IMG_H, IMG_W])
                print('NO IMAGE')
            if self.use_audio:
                print(data.keys())
                wav = data['wav'][i]
                wav = np.reshape(np.array(wav), [WAV_C, -1]) / 32768.0
                wavs.append(wav2freq(wav))
            doneA = data['done'][i][0]
            reward = data['reward'][i][0]
            _posA =  data['pos'][i]
            _posAL = data['goodpos'][i]
            _posAF = data['badpos'][i]
            _posL = list(data['campos'][i])
            _posL[0], _posL[1], _posL[2] = np.tanh(_posL[0] - 0.5), _posL[1], np.tanh(_posL[2])
            _posF = list(data['fakecampos'][i])
            _posF[0], _posF[1], _posF[2] = np.tanh(_posF[0] - 0.5), _posF[1], np.tanh(_posF[2])
            obj = str(data['obj'][i])
            color = str(data['color'][i])
            '''
            obj_oh = np.zeros([3, 10])
            obj_oh[COLOR_NAME.index(color)][OBJ_NAME.index(obj)] = 1.
            obj_oh = np.reshape(obj_oh, [30])
            '''
            obj_oh = np.zeros([self.observation_space['obj']])
            if obj != "None":
                obj_oh[OBJ_NAME.index(obj)] = 1.
            objs.append(obj_oh)
            img = np.reshape(np.array(img), [IMG_C, IMG_H, IMG_W]) / 255.0
            if img.shape[0] == 3 * IMG_C: #RGB -> G
                #print("Input is RGB, but using grayscale in setting. Converting...")
                temp_imgs = []
                for i in range(IMG_C):
                    temp_imgs.append(0.299 * img[3*i] + 0.587 * img[3*i+1] + 0.114 * img[3*i+2])
                img = np.stack(temp_imgs, axis = 0)
            imgs.append(img)
            rewards.append(reward)
            posAF.append(_posAF)
            posAL.append(_posAL)
            posA.append(_posA)
            posL.append(_posL)
            posF.append(_posF)
            if doneA: done.append(True)
            else: done.append(False)
        #info = (data['pos'], data['campos'])
        posAL = np.array(posAL)
        posAF = np.array(posAF)
        posA = np.array(posA)
        posL = np.array(posL)
        posF = np.array(posF)
        #print(np.linalg.norm(posAL, axis = 1))
        info = (posA, posL, posAL, posF, posAF)
        imgs = np.array(imgs)
        obs = {'image': imgs, 'obj': objs}
        if self.use_audio:
            obs['audio'] = np.array(wavs)
        if self.record:
            obs['imageT'] = imgsT
        return (obs, rewards, done, info)

    def step(self, action, ignore_agent_dim = True):
        super().send_action(action)
        return self.collect_observations(ignore_agent_dim)
