from db_query import DBQuery
import numpy as np
from utils import convert_list_to_dict
from dialogue_config import all_intents, all_slots, usersim_default_key
import copy
class StateTracker:
    """Tracks the state of the episode/conversation and prepares the state representation for the agent."""

    def __init__(self, database, constants):
        """
        The constructor of StateTracker.

        The constructor of StateTracker which creates a DB query object, creates necessary state rep. dicts, etc. and
        calls reset.

        Parameters:
            database (dict): The database with format dict(long: dict)
            constants (dict): Loaded constants in dict

        """

        self.db_helper = DBQuery(database)
        self.match_key = usersim_default_key
        self.intents_dict = convert_list_to_dict(all_intents)
        self.num_intents = len(all_intents)
        self.slots_dict = convert_list_to_dict(all_slots)
        self.num_slots = len(all_slots)
        self.max_round_num = constants['run']['max_round_num']
        self.none_state = np.zeros(self.get_state_size())
        self.p = 0
        #self.reward = 0
        #self.ga = 0
        #self.cum_reward = 0
        #self.unavslt = []
        P = {}
        self.reset(P,0)

    def get_state_size(self):
        """Returns the state size of the state representation used by the agent."""

        return 2 * self.num_intents + 8 * self.num_slots + 31

    def reset(self,P,ps):
        """Resets current_informs, history and round_num."""

        self.current_informs = {}
        # A list of the dialogues (dicts) by the agent and user so far in the conversation
        self.history = []
        self.Persona = P       ##Persona KnowledgeBase
        self.PS = []           ##Persona Match Tracker
        self.round_num = 0
        self.ga = 0            ## Goal Avalability Signal
        self.reward = 0
        self.cum_reward = 0
        self.unav_slt = []
        self.av_slt=[]
        self.p = 0             #Persuasion Stage
        self.sent = 0.5
        self.PSt = ps
        self.str = [1,0,0,0,0]

        if ps==0 or ps==3:
            self.str[0] = 1
        elif ps==1:
            self.str[1] = 1
        elif ps==2:
            self.str[2] = 1
        elif ps==4:
            self.str[3] = 1
        elif ps==5:
            self.str[4] = 1

        self.strategy = {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'LogicalAppeal',4:'PersonalAppeal',5:'Persuasion'}

    def print_history(self):
        """Helper function if you want to see the current history action by action."""

        for action in self.history:
            print(action)

    def get_state(self, done=False):
        """
        Returns the state representation as a numpy array which is fed into the agent's neural network.

        The state representation contains useful information for the agent about the current state of the conversation.
        Processes by the agent to be fed into the neural network. Ripe for experimentation and optimization.

        Parameters:
            done (bool): Indicates whether this is the last dialogue in the episode/conversation. Default: False

        Returns:
            numpy.array: A numpy array of shape (state size,)

        """

        # If done then fill state with zeros
        if done:
            return self.none_state,[]

        user_action = self.history[-1]
        db_results_dict = self.db_helper.get_db_results_for_slots(self.current_informs)
        last_agent_action = self.history[-2] if len(self.history) > 1 else None

        # Create one-hot of intents to represent the current user action
        user_act_rep = np.zeros((self.num_intents,))
        user_act_rep[self.intents_dict[user_action['intent']]] = 1.0

        # Create bag of inform slots representation to represent the current user action
        user_inform_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['inform_slots'].keys():
            user_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of request slots representation to represent the current user action
        user_request_slots_rep = np.zeros((self.num_slots,))
        for key in user_action['request_slots'].keys():
            user_request_slots_rep[self.slots_dict[key]] = 1.0

        # Create bag of filled_in slots based on the current_slots
        current_slots_rep = np.zeros((self.num_slots,))
        for key in self.current_informs:
            current_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent intent
        agent_act_rep = np.zeros((self.num_intents,))
        if last_agent_action:
            agent_act_rep[self.intents_dict[last_agent_action['intent']]] = 1.0

        # Encode last agent inform slots
        agent_inform_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['inform_slots'].keys():
                agent_inform_slots_rep[self.slots_dict[key]] = 1.0

        # Encode last agent request slots
        agent_request_slots_rep = np.zeros((self.num_slots,))
        if last_agent_action:
            for key in last_agent_action['request_slots'].keys():
                agent_request_slots_rep[self.slots_dict[key]] = 1.0

        # Value representation of the round num
        turn_rep = np.zeros((1,)) + self.round_num / 5.

        # One-hot representation of the round num
        turn_onehot_rep = np.zeros((self.max_round_num,))
        turn_onehot_rep[self.round_num - 1] = 1.0

        # Representation of DB query results (scaled counts)
        kb_count_rep = np.zeros((self.num_slots + 1,)) + db_results_dict['matching_all_constraints'] / 100.
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_count_rep[self.slots_dict[key]] = db_results_dict[key] / 100.

        # Representation of DB query results (binary)
        kb_binary_rep = np.zeros((self.num_slots + 1,)) + np.sum(db_results_dict['matching_all_constraints'] > 0.)
        for key in db_results_dict.keys():
            if key in self.slots_dict:
                kb_binary_rep[self.slots_dict[key]] = np.sum(db_results_dict[key] > 0.)


        ##GDM and Rewaed
        goal_pre = self.ga
        slt_status = np.zeros((self.num_slots,))
        for key in self.unav_slt:
            slt_status[self.slots_dict[key]] = -1
        for key in self.av_slt:
            slt_status[self.slots_dict[key]] = 1
            if key in self.PS:
                slt_status[self.slots_dict[key]] = 2
        per = self.p
        reward = self.reward
        cum_reward = self.cum_reward
        senti = self.sent
        PSt = self.str






        state_info = [turn_rep,goal_pre,per,self.av_slt,self.unav_slt,self.PS,slt_status,self.PSt]

        state_representation = np.hstack(
            [user_act_rep, user_inform_slots_rep, user_request_slots_rep, agent_act_rep, agent_inform_slots_rep,
             agent_request_slots_rep, current_slots_rep, turn_rep, turn_onehot_rep, kb_binary_rep,
             kb_count_rep,goal_pre,per,PSt,slt_status,reward]).flatten()



        return state_representation,state_info

    def update_state_agent(self, agent_action,ui):
        """
        Updates the dialogue history with the agent's action and augments the agent's action.

        Takes an agent action and updates the history. Also augments the agent_action param with query information and
        any other necessary information.

        Parameters:
            agent_action (dict): The agent action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'Agent')

        """
        available_result = {}
        url = ' '
        if agent_action['intent'] == 'inform':
            assert agent_action['inform_slots']
            uas,inform_slots,a,available_result,url = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs,ui)
            #print('Update STATE_agent current_informs',self.current_informs)
            print('Avalability: ',a)
            if(a==0):
                print('Unavalable Slots:',uas)
                self.goal_unavalabilty_P()
            agent_action['inform_slots'] = inform_slots
            # print(agent_action['inform_slots'])
            # message = client.messages.create(
            #                   from_='whatsapp:+14155238886',
            #                   body=str(agent_action['inform_slots']),
            #                   to='whatsapp:+918017153137'
            #               )
            assert agent_action['inform_slots']
            key, value = list(agent_action['inform_slots'].items())[0]  # Only one
            assert key != 'match_found'
            #print('val:',value)
            # message = client.messages.create(
            #                   from_='whatsapp:+14155238886',
            #                   body='PLACEHOLDER:'+str(value),
            #                   to='whatsapp:+918017153137'
            #               )
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key)
            #self.current_informs[key] = value
        # If intent is match_found then fill the action informs with the matches informs (if there is a match)
        elif agent_action['intent']=='Persuasion':
            if(self.ga):
                self.p = self.p + 1
            uas,inform_slots,a,available_result,url = self.db_helper.fill_inform_slot(agent_action['inform_slots'], self.current_informs,ui)
            agent_action['inform_slots'] = inform_slots
            assert agent_action['inform_slots']
            key, value = list(agent_action['inform_slots'].items())[0]  # Only one
            agent_action['inform_slots'][key] = value
            assert key != 'match_found'
            assert value != 'PLACEHOLDER', 'KEY: {}'.format(key)


        elif agent_action['intent'] in list(self.strategy.values()):
            if(self.ga):
                self.p = self.p + 1
            uas,avs,db_results,ids,a,url = self.db_helper.get_db_results(self.current_informs,ui)
            available_result = db_results

        elif agent_action['intent']=='Repersuasion' and self.p >= 2:
            self.p = self.p + 1
        elif agent_action['intent'] == 'match_found':
            assert not agent_action['inform_slots'], 'Cannot inform and have intent of match found!'
            uas,avs,db_results,ids,a,url = self.db_helper.get_db_results(self.current_informs,ui)
            available_result = db_results
            #uas,avs,db_results,ids,a = self.db_helper.get_db_results(self.current_informs,ui)
            if(a==0):
                print('Unavalable Slots:',uas)
                self.goal_unavalabilty_P()

            # if(a==0):
            #     self.goal_unavalabilty_P()
            # else:
            #     self.ga = 0
            if db_results:
                # Arbitrarily pick the first value of the dict
                key, value = list(db_results.items())[0]
                #print('SELECTED: ',key)
                agent_action['inform_slots'] = copy.deepcopy(value)
                agent_action['inform_slots'][self.match_key] = str(key)
            else:
                agent_action['inform_slots'][self.match_key] = 'no match available'
            #self.current_informs[self.match_key] = agent_action['inform_slots'][self.match_key] ****
        agent_action.update({'round': self.round_num, 'speaker': 'Agent'})
        self.history.append(agent_action)
        if agent_action['intent'] == 'inform':
            return value,available_result,url
        else:
            return -1,available_result,url

    def update_state_user(self, user_action,reward,sent):
        """
        Updates the dialogue history with the user's action and augments the user's action.

        Takes a user action and updates the history. Also augments the user_action param with necessary information.

        Parameters:
            user_action (dict): The user action of format dict('intent': string, 'inform_slots': dict,
                                 'request_slots': dict) and changed to dict('intent': '', 'inform_slots': {},
                                 'request_slots': {}, 'round': int, 'speaker': 'User')

        """
        if(type(user_action)==tuple):
            user_action = user_action[1]
        for key, value in user_action['inform_slots'].items():
            current_check  = {}
            if(value!='anything' and value!='Y'):
                self.current_informs[key] = value
        uas,avs,a,b,c,url = self.db_helper.get_db_results(self.current_informs,1)

        if(len(avs)):
            self.av_slt = self.av_slt + avs
            self.av_slt = list(set(self.av_slt))
        for element in avs:
            if element in self.Persona:
                if self.current_informs[element]==self.Persona[element]:
                    self.PS = self.PS + [element]

        print('Update State w.r.t User')
        print('Avalability:',c)
        if(c==0):
            print('Unavalable Slots:',uas)
            self.unav_slt = self.unav_slt + uas
            self.unav_slt = list(set(self.unav_slt))
            self.goal_unavalabilty_P()
            assert uas, "Unavalability but Unavalable Slots empty"


        else:
            self.ga = 0

        if(self.ga and self.p>0 and user_action['intent']!='reject'):
            self.ga = 0
        # print('unavslt:',self.unav_slt)
        user_action.update({'round': self.round_num, 'speaker': 'User'})
        self.history.append(user_action)
        self.reward = reward
        self.cum_reward = self.cum_reward + reward/10
        self.round_num += 1
        self.sent = sent

    def goal_unavalabilty_P(self):
        if self.ga == 0:
            self.ga = self.ga + 1

    def psupdate(self,ps):
        self.p = ps
        self.PSt = ps
        self.str = [1,0,0,0,0]
        if ps==0 or ps==3:
            self.str[0] = 1
        elif ps==1:
            self.str[1] = 1
        elif ps==2:
            self.str[2] = 1
        elif ps==4:
            self.str[3] = 1
        elif ps==5:
            self.str[4] = 1
