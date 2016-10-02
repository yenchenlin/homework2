import tensorflow as tf
import numpy as np

class CategoricalPolicy(object):
    def __init__(self, in_dim, out_dim, hidden_dim, optimizer, session):

        # Placeholder Inputs
        self._observations = tf.placeholder(tf.float32, shape=[None, in_dim], name="observations")
        self._actions = tf.placeholder(tf.int32, name="actions")
        self._advantages = tf.placeholder(tf.float32, name="advantages")

        self._opt = optimizer
        self._sess = session

        # 1. Use TensorFlow to construct a 2-layer neural network as stochastic policy
        #    Each layer should be fully-connected and have size `hidden_dim`
        #    Use tanh as the activation function of the first hidden layer, and append softmax layer after the output
        #    of the neural network to get the probability of each possible action
        #
        # 2. Assign the output of the softmax layer to the variable `probs`
        #    Let's assume n_batch equals to `self._observations.get_shape()[0]`,
        #    then shape of the variable `probs` should be [n_batch, n_actions]
        
        # YOUR CODE HERE >>>>>>
        # <<<<<<<<

        # --------------------------------------------------
        # This operation (variable) is used when choosing action during data sampling phase
        # Shape of probs: [1, n_actions]
        
        act_op = probs[0, :]

        # --------------------------------------------------
        # Following operations (variables) are used when updating model
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

        # --------------------------------------------------
        # This operation (variable) is used when choosing action during data sampling phase
        self._act_op = act_op
        
        # --------------------------------------------------
        # These operations (variables) are used when updating model
        self._loss_op = surr_loss
        self._train_op = train_op

    def act(self, observation):
        # expect observation to be of shape [1, observation_space]
        action_probs = self._sess.run(self._act_op, feed_dict={self._observations: observation})
        
        # `action_probs` is an array that has shape [1, action_space], it contains the probability of each action
        # Your code should sample and return an action (i.e., 0 or 1) according to `action_probs`
        
        # YOUR CODE HERE >>>>>>
        # <<<<<<<<

    def train(self, observations, actions, advantages):
        loss, _ = self._sess.run([self._loss_op, self._train_op], feed_dict={self._observations:observations, self._actions:actions, self._advantages:advantages})
        return loss