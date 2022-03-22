import gym
import pygame
env = gym.make("CartPole-v1")
obs = env.reset()

print(obs)

img= env.render(mode="rgb_array")
print(env.action_space) #정수 0과 1 사이의 가능한 행동 파악하기

action = 1 #오른쪽으로 가속
obs, reward, done, info = env.step(action)
'''
obs: 새로운 관측값.
reward: 어떤 행동을 시행해도 매 스텝마다 1.0 보상. -> 목적: 가능한한 오랫동안 실행하는 것.
done: 이 값이 True면 에피소드가 끝난 것. -> 초기화 후 다시 사용.
info: 디버깅이나 훈련에 유용한 추가적인 정보가 담길 수 있음.
'''
#500번 실행
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals =[]
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
        totals.append(episode_rewards)
        
import numpy as np
print("\nmean, std, min, max:\n", np.mean(totals), np.std(totals), np.min(totals), np.max(totals))

#할인 계수(gamma) 가 0에 가까우면 미래의 보상은 현재의 보상만큼 중요하지 않음.
#반대로 1에 가까우면 미래의 보상이 현의 보상보다 중요함.

import tensorflow as tf
from tensorflow import keras

def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads









