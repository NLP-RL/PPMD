# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import random, copy
import numpy as np
from dialogue_config import rule_requests, agent_actions
import re
#from utils import graph
import tensorflow as tf
graph = tf.get_default_graph()

#-----
import tensorflow as tf
from tensorflow.python.keras import backend as K
#from BERT_NLU import graph
#from BERT_NLU import sess
# graph = tf.get_default_graph()
# # from bottom copied and pasted above
# sess = tf.compat.v1.Session()
# K.set_session

class DQNAgent:
    def __init__(self, state_size, constants):
        """
        Parameters:
            state_size (int): The state representation size or length of numpy array
            constants (dict): Loaded constants in dict

        """

        self.C = constants['agent']
        self.memory = []
        self.memory_index = 0
        self.max_memory_size = self.C['max_mem_size']
        self.eps = self.C['epsilon_init']
        self.min_eps = self.C['min_epsilon']
        self.vanilla = self.C['vanilla']
        self.lr = self.C['learning_rate']
        self.gamma = self.C['gamma']
        self.batch_size = self.C['batch_size']
        self.hidden_size = self.C['dqn_hidden_size']

        self.load_weights_file_path = self.C['load_weights_file_path']
        self.save_weights_file_path = self.C['save_weights_file_path']

        if self.max_memory_size < self.batch_size:
            raise ValueError('Max memory size must be at least as great as batch size!')

        self.state_size = state_size
        self.possible_actions = agent_actions
        self.num_actions = len(self.possible_actions)

        self.rule_request_set = rule_requests

        self.beh_model = self._build_model()
        self.tar_model = self._build_model()
        self._load_weights()

        self.reset()

    def _build_model(self):
        """Builds and returns model/graph of neural network."""

        model = Sequential()
        model.add(Dense(self.hidden_size, input_dim=self.state_size, activation='relu'))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.lr))
        return model

    def reset(self):
        """Resets the rule-based variables."""

        self.rule_current_slot_index = 0
        self.rule_phase = 'not done'

    def get_action(self, state, use_rule=False,train=True):

        """

        Parameters:
            state (numpy.array): The database with format dict(long: dict)
            use_rule (bool): Indicates whether or not to use the rule-based policy, which depends on if this was called
                             in warmup or training. Default: False

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself

        """
        return self._dqn_action(state)


    def _map_action_to_index(self, response):
        """
        Maps an action to an index from possible actions.

        Parameters:
            response (dict)

        Returns:
            int
        """

        for (i, action) in enumerate(self.possible_actions):
            if response == action:
                return i
        raise ValueError('Response: {} not found in possible actions'.format(response))

    def _dqn_action(self, state):
        """
        Returns a behavior model output given a state.

        Parameters:
            state (numpy.array)

        Returns:
            int: The index of the action in the possible actions
            dict: The action/response itself
        """

        index = np.argmax(self._dqn_predict_one(state))
        action = self._map_index_to_action(index)
        return index, action

    def _map_index_to_action(self, index):
        """
        Maps an index to an action in possible actions.

        Parameters:
            index (int)

        Returns:
            dict
        """

        for (i, action) in enumerate(self.possible_actions):
            if index == i:
                return copy.deepcopy(action)
        raise ValueError('Index: {} not in range of possible actions'.format(index))

    def _dqn_predict_one(self, state, target=False):
        """
        Returns a model prediction given a state.

        Parameters:
            state (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        return self._dqn_predict(state.reshape(1, self.state_size), target=target).flatten()

    def _dqn_predict(self, states, target=False):
        """
        Returns a model prediction given an array of states.

        Parameters:
            states (numpy.array)
            target (bool)

        Returns:
            numpy.array
        """

        if target:
            # global graph
            # global sess
            with graph.as_default():
                return self.tar_model.predict(states)
        else:
            # global graph
            # global sess
            # with graph.as_default():
            #     K.set_session(sess)
            with graph.as_default():
                return self.beh_model.predict(states)
    def _load_weights(self):
        """Loads the weights of both models from two h5 files."""

        if not self.load_weights_file_path:
            return
        global sess
        #K.set_session(sess)#ashok added this line
        beh_load_file_path = re.sub(r'\.h5', r'_beh.h5', self.load_weights_file_path)
        self.beh_model.load_weights(beh_load_file_path)
        tar_load_file_path = re.sub(r'\.h5', r'_tar.h5', self.load_weights_file_path)
        self.tar_model.load_weights(tar_load_file_path)
