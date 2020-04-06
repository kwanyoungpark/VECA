#from model import *
import numpy as np
from utils import wav2freq
from VECA.environment import GeneralEnvironment

IMG_H, IMG_W = 84, 84

class Environment(GeneralEnvironment):
	def __init__(self, NUM_AGENTS, port = 8970):
		GeneralEnvironment.__init__(self, NUM_AGENTS, port)
	
	def step(self, action):
		data = super().step(action)
		rewards, done, info = [], [], []
		imgs, wavs = [], []
		for i in range(self.num_envs):
			#print('got observation':)
			#img = list(reversed(readBytes(data[:IMG_DATAL], 'uint8')))
			#wav = readBytes(data[IMG_DATAL:IMG_DATAL + 2 * WAV_DATAL], 'int16')
			img = list(reversed(data['img'][i]))
			wav = data['wav'][i]
			doneA = data['done'][i][0]
			#other = data['other'][i][0]
			reward = data['reward'][i][0]
			pos = data['pos'][i]
			#pos = readBytes(data[IMG_DATAL + 2 * WAV_DATAL + 6:], 'float')
			#print('pos', pos)
			#print(data[-50:])
			img = np.reshape(np.array(img), [2, IMG_H, IMG_W]) / 255.0
			wav = np.reshape(np.array(wav), [2, -1]) / 32768.0 
			wav = wav2freq(wav)
			#other = np.reshape(np.array(other).astype(np.float32), [1])
			#print(np.min(wav), np.max(wav))
			imgs.append(img)
			wavs.append(wav)
			#others.append(other)
			rewards.append(reward)
			if doneA: done.append(True)
			else: done.append(False)
			info.append(pos)
		imgs, wavs = np.array(imgs), np.array(wavs)
		obs = (imgs, wavs)
		return (obs, rewards, done, info)


