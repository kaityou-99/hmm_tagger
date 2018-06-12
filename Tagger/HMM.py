#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division

import numpy as np
from sklearn.preprocessing import LabelEncoder
import functools

def str_to_list(func):
    @functools.wraps(func)
    def wrapper(instance, input, **kwargs):
        if isinstance(input, str):
            ret = func(instance, [input])[0]
        elif hasattr(input, "__iter__"):
            ret = func(instance, input)
        else:
            raise AttributeError("unexpected argument was passed.")
        return ret
    return wrapper


class PlainHMM(object):

    def __init__(self):
        self._le_state2idx = LabelEncoder()
        self._le_repr2idx = LabelEncoder()
        self._set_state = None
        self._init_state = None
        self._set_repr = None
        self._mat_state_trans_prob = None
        self._mat_repr_obs_prob = None

    def _validate_state_trans_prob(self):
        assert self._mat_state_trans_prob is not None, "state transition probability is not initialized yet."
        assert all(np.abs(self._mat_state_trans_prob.sum(axis=1)-1) < 1E-4), "sum of transition probability must be equal to 1."

    def _validate_repr_obs_prob(self):
        assert self._mat_repr_obs_prob is not None, "representation probability is not initialized yet."
        assert all(np.abs(self._mat_repr_obs_prob.sum(axis=1)-1) < 1E-4), "sum of representation probability must be equal to 1."

    @str_to_list
    def _state2index(self, lst_state):
        return self._le_state2idx.transform(lst_state)

    @str_to_list
    def _repr2index(self, lst_repr):
        return self._le_repr2idx.transform(lst_repr)

    @str_to_list
    def _index2state(self, lst_index):
        return self._le_state2idx.inverse_transform(lst_index)

    @str_to_list
    def _index2repr(self, lst_index):
        return self._le_repr2idx.inverse_transform(lst_index)

    @property
    def n_state(self):
        return len(self._set_state)

    @property
    def n_repr(self):
        return len(self._set_repr)

    def set_state(self, states, state_bos):
        assert hasattr(states, "__iter__"), "argument `states` must be iterable."
        # store set of state symbol(e.g. PoS tag)
        self._set_state = set(states)
        self._n_state = len(self._set_state)
        assert state_bos in self._set_state, "argument `state_bos` must be an element of `states`."
        self._init_state = state_bos

        # create mapper between symbol and index
        self._le_state2idx.fit(states)

    def set_repr(self, representations):
        assert hasattr(representations, "__iter__"), "argument `representations` must be interable."
        # store set of representation symbol(e.g. word)
        self._set_repr = set(representations)
        self._n_repr = len(self._set_repr)

        # create mapper between symbol and index
        self._le_repr2idx.fit(representations)

    def set_state_trans_prob(self, lst_tuple_trans_prob):
        assert hasattr(lst_tuple_trans_prob, "__iter__"), "argument `lst_tuple_trans_prob` must be iterable."
        assert all(np.array(list(map(len, lst_tuple_trans_prob))) == 3), "argument `lst_tuple_trans_prob` must be the tuple of (s_prev, s_next, trans_prob)"

        self._mat_state_trans_prob = np.zeros(shape=(self._n_state, self._n_state))
        for s_0, s_1, p in lst_tuple_trans_prob:
            i = self._state2index(s_0)
            j = self._state2index(s_1)
            self._mat_state_trans_prob[i,j] = p
        self._mat_log_state_trans_prob = np.log(self._mat_state_trans_prob)

    def set_repr_obs_prob(self, lst_tuple_repr_prob):
        attr_name = "lst_tuple_trans_prob"
        assert hasattr(lst_tuple_repr_prob, "__iter__"), "argument `%s` must be iterable." % attr_name
        assert all(np.array(list(map(len, lst_tuple_repr_prob))) == 3), "argument `%s` must be the tuple of (state, repr, obs_prob)" % attr_name

        self._mat_repr_obs_prob = np.zeros(shape=(self._n_state, self._n_repr))
        for s, r, p in lst_tuple_repr_prob:
            i = self._state2index(s)
            j = self._repr2index(r)
            self._mat_repr_obs_prob[i,j] = p
        self._mat_log_repr_obs_prob = np.log(self._mat_repr_obs_prob)

    def evaluate(self, states, representations, verbose=False):
        if states[0] != self._init_state:
            states = [self._init_state] + states
        assert len(states) == len(representations)+1, "length mismatch detected."

        arry_state_idx = self._state2index(states)
        arry_repr_idx = self._repr2index(representations)

        nll = 0.
        for s_0, s_1, r in zip(arry_state_idx[:-1], arry_state_idx[1:], arry_repr_idx):
            if verbose:
                print("p(s_t+1|s_t): %f, p(r_t|s_t): %f" % (self._mat_state_trans_prob[s_0, s_1], self._mat_repr_obs_prob[s_1, r]))
            nll -= self._mat_log_state_trans_prob[s_0, s_1]
            nll -= self._mat_log_repr_obs_prob[s_1,r]

        return nll


class HMM(PlainHMM):

    def __init__(self):
        super(HMM, self).__init__()

    def predict(self, representations, verboes=False):

        arry_repr_idx = self._repr2index(representations)

        # find MAP path using viterbi algorithm
        n_t = arry_repr_idx.size
        n_s = self.n_state

        s_m1 = self._state2index(self._init_state)
        vec_ll_prev = np.zeros(n_s)
        vec_ll_next = np.empty(n_s)
        mat_trace_trans = np.empty((n_s, n_t), dtype=np.int32)
        for t in np.arange(n_t):
            r_t = arry_repr_idx[t]
            for s_t in np.arange(n_s):
                # p(s_t|r_{1:t}) = max{ p(s_t|s_t-1)*p(r_t|s_t)*p(s_t-1|r_{1:t-1})}
                if t==0:
                    vec_candidates = vec_ll_prev + self._mat_log_state_trans_prob[s_m1,s_t] + self._mat_log_repr_obs_prob[s_t,r_t]
                else:
                    vec_candidates = vec_ll_prev + self._mat_log_state_trans_prob[:,s_t] + self._mat_log_repr_obs_prob[s_t,r_t]
                vec_ll_next[s_t] = np.max(vec_candidates)

                # phi(s_t,t) = argmax_{s_t-1}{p(s_t|s_t-1)*p(r_t|s_t)*p(s_t-1|r_{1:t-1})}
                if t==0:
                    mat_trace_trans[s_t,t] = s_m1
                else:
                    mat_trace_trans[s_t,t] = np.argmax(vec_candidates)
            # proceed to next step
            vec_ll_prev = vec_ll_next.copy()

        # trace backward
        arry_state_idx = np.empty(n_t, dtype=np.int32)
        nll = -np.max(vec_ll_next)
        for t in np.arange(start=n_t, stop=0, step=-1):
            if t == n_t:
                arry_state_idx[t-1] = np.argmax(vec_ll_next)
            else:
                s_t = arry_state_idx[t]
                arry_state_idx[t-1] = mat_trace_trans[s_t,t]

        lst_state_map = self._index2state(arry_state_idx).tolist()
        return nll, lst_state_map