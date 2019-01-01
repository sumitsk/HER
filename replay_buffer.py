# import threading
import numpy as np
import random

from baselines.common.segment_tree import SumSegmentTree, MinSegmentTree
import ipdb


class ReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sampler):
        """Creates a replay buffer.
        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
        """
        self.buffer_shapes = buffer_shapes
        self.size = size_in_transitions // T
        self.T = T
        self.sampler = sampler

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape])
                        for key, shape in buffer_shapes.items()}

        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0

        # self.lock = threading.Lock()

    @property
    def full(self):
        # with self.lock:
        return self.current_size == self.size

    def sample(self, batch_size):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        # with self.lock:
        assert self.current_size > 0
        for key in self.buffers.keys():
            buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        transitions = self.sampler.sample(buffers, batch_size)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        return transitions

    def store_episode(self, episode_batch):
        """episode_batch: array(batch_size x (T or T+1) x dim_key)
        """
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        # with self.lock:
        idxs = self._get_storage_idx(batch_size)

        # load inputs into buffers
        for key in self.buffers.keys():
            self.buffers[key][idxs] = episode_batch[key]

        self.n_transitions_stored += batch_size * self.T

    def get_current_episode_size(self):
        # with self.lock:
        return self.current_size

    def get_current_size(self):
        # with self.lock:
        return self.current_size * self.T

    def get_transitions_stored(self):
        # with self.lock:
        return self.n_transitions_stored

    def clear_buffer(self):
        # with self.lock:
        self.current_size = 0

    def _get_storage_idx(self, inc=None):
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)

        if inc == 1:
            idx = idx[0]
        return idx


class PrioritizedReplayBuffer:
    def __init__(self, buffer_shapes, size_in_transitions, T, sample_transitions, alpha=0.5):
        raise NotImplementedError('implement PrioritizedReplayBuffer')
        """Creates a replay buffer for prioritized sampling.

        Args:
            buffer_shapes (dict of ints): the shape for all buffers that are used in the replay
                buffer
            size_in_transitions (int): the size of the buffer, measured in transitions
            T (int): the time horizon for episodes
            sample_transitions (function): a function that samples from the replay buffer
            alpha (float): parameter for prioritized sampling
        """
        self.buffer_shapes = buffer_shapes
        # size in episodes
        self.size = size_in_transitions // T
        self.T = T
        self.sample_transitions = sample_transitions

        # self.buffers is {key: array(size_in_episodes x T or T+1 x dim_key)}
        self.buffers = {key: np.empty([self.size, *shape]) for key, shape in buffer_shapes.items()}

        # memory management
        # number of episodes stored in replay buffer
        self.current_size = 0
        # number of transitions stored in replay buffer (= size of segment tree)
        self.current_count = 0
        # total number of transitions so far
        self.n_transitions_stored = 0

        self.lock = threading.Lock()

        # Prioritized Experience Replay specific
        assert alpha >=0
        self._alpha = alpha

        it_capacity = 1
        max_transitions = self.size * T
        while it_capacity < max_transitions:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

        self.flatten_indices = np.arange(max_transitions).reshape(self.size, T)

    @property
    def full(self):
        with self.lock:
            return self.current_size == self.size

    def get_current_episode_size(self):
        with self.lock:
            return self.current_size

    def get_current_size(self):
        with self.lock:
            return self.current_size * self.T

    def get_transitions_stored(self):
        with self.lock:
            return self.n_transitions_stored

    def clear_buffer(self):
        with self.lock:
            self.current_size = 0
            self.current_count = 0

    def sample(self, batch_size, beta, obj=None):
        """Returns a dict {key: array(batch_size x shapes[key])}
        """
        buffers = {}

        with self.lock:
            assert self.current_size > 0
            for key in self.buffers.keys():
                buffers[key] = self.buffers[key][:self.current_size]

        buffers['o_2'] = buffers['o'][:, 1:, :]
        buffers['ag_2'] = buffers['ag'][:, 1:, :]

        # indices of all the samples to be replayed, some of these will be substituted by HER samples
        with self.lock:
            transition_indices = self._sample_proportional(batch_size)
        # convert transition indices to episode indices and timestep of that episode
        episode_idxs, t_samples = self._get_episode_and_time_indices(transition_indices)

        # sample transitions and return original and her samples 
        transitions, original_samples, her_samples = self.sample_transitions(buffers, batch_size, episode_idxs=episode_idxs,
                                                                             t_samples=t_samples, return_indices=True)

        for key in (['r', 'o_2', 'ag_2'] + list(self.buffers.keys())):
            assert key in transitions, "key %s missing from transitions" % key

        # weights of all the samples
        weights = np.ones(batch_size)
        
        # compute weights of original samples
        with self.lock:
            org_weights = self._compute_sample_weights(transition_indices[original_samples], beta)
            weights[original_samples] = org_weights
            if obj is not None:
                her_transitions = {key: value[her_samples] for key, value in transitions.items()}
                her_priorities = obj.get_priorities(her_transitions)
                her_weights = self._compute_her_sample_weights(her_priorities, beta)
                # NOTE: normalizing max her weight to be equal to max org weight
                her_weights = her_weights / max(her_weights) * max(org_weights)
                weights[her_samples] = her_weights
        return transitions, weights, (transition_indices, original_samples)

    def _compute_sample_weights(self, idxes, beta):
        # returns weights of all the original samples (not HER ones)

        weights = np.full(len(idxes), 1.0)
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * self.current_count) ** (-beta)
        for i, idx in enumerate(idxes):
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * self.current_count) ** (-beta)
            weights[i] = weight / max_weight
        return weights

    def _compute_her_sample_weights(self, priorities, beta):
        priorities = priorities ** self._alpha
        probs = priorities / sum(priorities)
        # TODO: is self.current_count right here ?
        weights = (probs * self.current_count) ** (-beta)
        weights = weights / max(weights)
        return weights

    def store_episode(self, episode_batch):
        # episode_batch: array(batch_size x (T or T+1) x dim_key)
        batch_sizes = [len(episode_batch[key]) for key in episode_batch.keys()]
        assert np.all(np.array(batch_sizes) == batch_sizes[0])
        batch_size = batch_sizes[0]

        with self.lock:
            idxs = self._get_storage_idx(batch_size)

            # load inputs into buffers
            for key in self.buffers.keys():
                self.buffers[key][idxs] = episode_batch[key]

            self.n_transitions_stored += batch_size * self.T

            # update tree for all the new additions in the buffer
            new_indices = self._get_new_flatten_indices(idxs)
            priority = self._max_priority ** self._alpha
            for new_idx in new_indices:
                self._it_sum[new_idx] = priority
                self._it_min[new_idx] = priority

    def _get_storage_idx(self, inc=None):
        # return episode incides where the incoming transitions will be stored
        inc = inc or 1   # size increment
        assert inc <= self.size, "Batch committed to replay is too large!"
        # go consecutively until you hit the end, and then go randomly.
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)

        # update replay size
        self.current_size = min(self.size, self.current_size+inc)
        self.current_count = min(self.size*self.T, self.current_size*self.T)

        if inc == 1:
            idx = idx[0]
        return idx

    def update_priorities(self, idxes_tuple, priorities):
        """Update priorities of sampled transitions.

        sets priority of transition at index idxes[i] in buffer to priorities[i].

        Parameters
        ----------
        idxes: tuple
            List of idxes of sampled transitions, list of indices which are original samples (not HER)
        priorities: [float]
            List of updated priorities corresponding to transitions at the sampled idxes denoted by variable `idxes`.
        """
        priorities = priorities.squeeze()
        idxes, org_indexes = idxes_tuple
        assert len(idxes) == len(priorities)
        # update the priority for the original indexes only (as they were the ones used in the replay)
        # the remaining ones are HER transitions (goal-substituted ones) which are not present in the experience replay
        org_idxes = idxes[org_indexes]
        org_priorities = priorities[org_indexes]
        with self.lock:
            count = self.current_count
            for idx, priority in zip(org_idxes, org_priorities):
                assert priority > 0
                assert 0 <= idx < count
                self._it_sum[idx] = priority ** self._alpha
                self._it_min[idx] = priority ** self._alpha

                self._max_priority = max(self._max_priority, priority)

    def _sample_proportional(self, batch_size):
        # prioritized sampling, returns transitions indices
        res = np.full(batch_size, 0)
        p_total = self._it_sum.sum(0, self.current_count - 1)
        every_range_len = p_total / batch_size
        for i in range(batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res[i] = idx
        return res

    def _get_new_flatten_indices(self, row_indices):
        return self.flatten_indices[row_indices].flatten()

    def _get_episode_and_time_indices(self, idxs):
        episode_idxs, t_samples = np.unravel_index(idxs, (self.size, self.T))
        return episode_idxs, t_samples