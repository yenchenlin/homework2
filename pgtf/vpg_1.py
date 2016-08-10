"""
Policy Gradients
How we optimize the policy
--------------------------
Types of parameterized policies
-------------------------------
Map s (state) to an output vector u
1. If the action is from a discrete set, the network maps s to a vector of probabilities (softmax)
2. If the action is continuous, then we map s to the mean/variance of a Gaussian distribution
(diagonal covariance that does not depend on s)
3. If a is binary valued, we use a single output, the probability of outputting 1 (although
we could also just use 1.)
TODO: implement baseline
TODO: implement generalized advantage estimation
"""
from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

from six.moves import range
from gym.spaces import Box, Discrete
from scipy.signal import lfilter

import gym
import tensorflow as tf
import numpy as np
import argparse
from pgtf.baselines.linear_feature_baseline import LinearFeatureBaseline

def flatten_space(space):
    if isinstance(space, Box):
        return np.prod(space.shape)
    elif isinstance(space, Discrete):
        return space.n
    else:
        raise ValueError("Env must be either Box or Discrete.")

def discount_cumsum(x, discount_rate):
    return lfilter([1], [1, -discount_rate], x[::-1], axis=0)[::-1]


class CategoricalPolicy(object):
    def __init__(self, in_dim, out_dim, hidden_dim, optimizer, session):

        # Placeholder Inputs
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")

        self._opt = optimizer
        self._sess = session

        h1 = tf.contrib.layers.fully_connected(self._observations, hidden_dim, activation_fn=tf.tanh)
        probs = tf.contrib.layers.fully_connected(h1, out_dim, activation_fn=tf.nn.softmax)

        # Used when choosing action during data sampling phase

        # Shape of probs: [1, n_actions]
        act_op = probs[0, :]

        # --------------------------------------------------
        # Used when updating model

        # Shape of probs: [total_timestep_iter, n_actions]

        # 1. Find first action index of each timestep in flattened vector form
        action_idxs_flattened = tf.range(0, tf.shape(probs)[0]) * tf.shape(probs)[1]

        # 2. Add index of action chosen at each timestep, so now
        # action_idxs_flattened represent action index chosen at each timestep
        action_idxs_flattened += self._actions

        # 3. Gather the probability of action at each timestep
        probs_vec = tf.gather(tf.reshape(probs, [-1]), action_idxs_flattened)

        log_lik = tf.log(probs_vec + 1e-8)

        surr_loss = -tf.reduce_mean(log_lik * self._advantages, name="loss_op")

        grads_and_vars = self._opt.compute_gradients(surr_loss)
        train_op = self._opt.apply_gradients(grads_and_vars, name="train_op")

        self._act_op = act_op
        self._loss_op = surr_loss
        self._train_op = train_op

    def act(self, observation):
        # expect observation to be shape(1, self.observation_space)
        a = self._sess.run(self._act_op, feed_dict={self._observations: observation})

        # Sample index of numpy array, probability of each index i is weighted
        # by array[i]
        cs = np.cumsum(a)
        idx = sum(cs < np.random.rand())

        return idx

    def train(self, observations, actions, advantages):
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict={self._observations:observations, self._actions:actions, self._advantages:advantages})
        return loss


class PolicyOptimizer(object):
    def __init__(self, env, policy, baseline, n_iter, n_episode, path_length,
        discount_rate=.99):

        self.policy = policy
        self.baseline = baseline
        self.env = env
        self.n_iter = n_iter
        self.n_episode = n_episode
        self.path_length = path_length
        self.discount_rate = discount_rate

    def sample_path(self):
        obs = []
        actions = []
        rewards = []
        ob = self.env.reset()

        for _ in range(self.path_length):
            a = self.policy.act(ob.reshape(1, -1))
            next_ob, r, done, _ = self.env.step(a)
            obs.append(ob)
            actions.append(a)
            rewards.append(r)
            ob = next_ob
            if done:
                break

        return dict(
            observations=np.array(obs),
            actions=np.array(actions),
            rewards=np.array(rewards),
        )

    def process_paths(self, paths):
        for p in paths:
            if self.baseline != None:
                b = self.baseline.predict(p)
            else:
                b = 0

            r = discount_cumsum(p["rewards"], self.discount_rate)
            a = r - b

            p["returns"] = r
            p["baselines"] = b
            p["advantages"] = (a - a.mean()) / (a.std() + 1e-8) # normalize

        obs = np.concatenate([ p["observations"] for p in paths ])
        actions = np.concatenate([ p["actions"] for p in paths ])
        rewards = np.concatenate([ p["rewards"] for p in paths ])
        advantages = np.concatenate([ p["advantages"] for p in paths ])

        return dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
        )


    def train(self):

        for i in range(1, self.n_iter+1):
            paths = []
            for _ in range(self.n_episode):
                paths.append(self.sample_path())
            data = self.process_paths(paths)
            loss = self.policy.train(data["observations"], data["actions"], data["advantages"])
            avg_return = np.mean([sum(p["rewards"]) for p in paths])
            print("Iteration {}: Loss = {}, Average Return = {}".format(i, loss, avg_return))
            if avg_return >= 195:
                print("Solve at {} iterations, whih equals {} episodes.".format(i, i*100))
                break

            if self.baseline != None:
                self.baseline.fit(paths)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_iter', default=200, type=int, help='number of iterations')
    parser.add_argument('--n_episode', default=100, type=int, help='number of episodes/iteration')
    parser.add_argument('--path_length', default=200, type=int, help='number of steps')
    parser.add_argument('--learning_rate', default=0.01, help='learning rate for Adam Optimizer')
    parser.add_argument('--discount_rate', default=0.99, help='discount rate of agent')
    parser.add_argument('--env', default='CartPole-v0', help='gym environment for training')
    parser.add_argument('--algorithm', default='VPG', help='algorithm identifier')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    env = gym.make(args.env)

    sess = tf.Session()

    in_dim = flatten_space(env.observation_space)
    out_dim = flatten_space(env.action_space)
    hidden_dim = 8

    opt = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
    policy = CategoricalPolicy(in_dim, out_dim, hidden_dim, opt, sess)
    baseline = LinearFeatureBaseline(None)
    po = PolicyOptimizer(env, policy, baseline, args.n_iter, args.n_episode, args.path_length,
                         args.discount_rate)

    sess.run(tf.initialize_all_variables())

    # train the policy optimizer
    po.train()
