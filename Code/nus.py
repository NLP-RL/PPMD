from dialogue_config import usersim_default_key, FAIL, NO_OUTCOME, SUCCESS, usersim_required_init_inform_keys,no_query_keys
from utils import reward_function, PMeR
import random, copy
from Template_NLG import NLG

class UserSimulator:
    """Simulates a real user, to train the agent with reinforcement learning."""

    def __init__(self, goal_list, constants, database):
        """
        Parameters:
            goal_list (list): User goals loaded from file
            constants (dict): Dict of constants loaded from file
            database (dict): The database in the format dict(long: dict)
        """

        self.goal_list = goal_list
        self.nlg = NLG()
        self.max_round = constants['run']['max_round_num']
        self.default_key = usersim_default_key
        # A list of REQUIRED to be in the first action inform keys
        self.init_informs = usersim_required_init_inform_keys
        self.no_query = no_query_keys


        self.database = database

    def reset(self,P,ps):
        """
        Resets the user sim. by emptying the state and returning the initial action.

        Returns:
            dict: The initial action of an episode
        """

        self.goal = random.choice(self.goal_list)
        self.p = 1
        self.am = {}
        print('Goal:',self.goal)
        self.gu = {}
        self.dgs = {}
        self.info = {}
        self.f = 0
        self.perno = 0
        self.req = {}
        self.str = [1,0,0,0,0]
        self.pmer = 0
        self.strategy = {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'LogicalAppeal',4:'PersonalAppeal',5:'Persuasion'}
        self.Pers = list(P.keys())
        self.ps = ps
        print('Personality Type:',self.strategy[self.ps])


        if(len(self.goal['fs'])):
            self.dgs.update(self.goal['inform_slots'])
            self.dgs.update(self.goal['fs'])
            self.am.update(self.goal['fs'])
        elif(len(self.goal['uu'])):
            self.p = 2
            self.gu.update(self.goal['inform_slots'])
            p = {}
            p.update(self.gu)
            self.gu.update(self.goal['uu'])
            self.am.update(self.goal['uu'])
            us = list(self.am.keys())[0]
            if us in list(p.keys()):
                p.pop(us)
            self.PMS = []
            self.GT = list(p.keys())
            if len(set(self.Pers).intersection(set(self.GT))):
                self.PMS = list(set(self.Pers).intersection(set(self.GT)))
            print('PMS:',self.PMS)
            print('GT:',self.GT)

            self.amk = list(self.am.keys())[0]
        if self.p==1:
                self.chh = self.dgs
        else:
            self.chh = self.gu

        self.reject  = 0
        self.info.update(self.goal['inform_slots'])
        self.req.update(self.goal['request_slots'])

        # Add default slot to requests of goal
        self.goal['request_slots'][self.default_key] = 'UNK'
        self.state = {}
        # Add all inform slots informed by agent or user sim to this dict
        self.state['history_slots'] = {}
        # Any inform slots for the current user sim action, empty at start of turn
        self.state['inform_slots'] = {}
        # Current request slots the user sim wants to request
        self.state['request_slots'] = {}
        # Init. all informs and requests in user goal, remove slots as informs made by user or agent
        self.state['rest_slots'] = {}
        self.state['rest_slots'].update(self.goal['inform_slots'])
        self.state['rest_slots'].update(self.goal['request_slots'])
        self.tot_slt = len(self.state['rest_slots'])
        #print('Intial Rest Slot: ',self.state['rest_slots'])
        self.state['intent'] = ''
        # False for failure, true for success, init. to failure
        self.constraint_check = FAIL
        self.sent = 0

        return self.tot_slt,self._return_init_action(),self.ps

    def _return_init_action(self):
        """
        Returns the initial action of the episode.

        The initial action has an intent of request, required init. inform slots and a single request slot.

        Returns:
            dict: Initial user response
        """

        # Always request
        self.state['intent'] = 'preq'

        # if self.goal['inform_slots']:
        #     # Pick all the required init. informs, and add if they exist in goal inform slots
        #     for inform_key in self.init_informs:
        #         if inform_key in self.goal['inform_slots']:
        #             self.state['inform_slots'][inform_key] = self.goal['inform_slots'][inform_key]
        #             self.state['rest_slots'].pop(inform_key)
        #             self.state['history_slots'][inform_key] = self.goal['inform_slots'][inform_key]
        #     # If nothing was added then pick a random one to add
        #     if not self.state['inform_slots']:
        #         key, value = random.choice(list(self.goal['inform_slots'].items()))
        #         self.state['inform_slots'][key] = value
        #         self.state['rest_slots'].pop(key)
        #         self.state['history_slots'][key] = value

        # Now add a request, do a random one if something other than def. available
        # self.goal['request_slots'].pop(self.default_key)
        # if self.goal['request_slots']:
        #     req_key = random.choice(list(self.goal['request_slots'].keys()))
        # else:
        #     req_key = self.default_key
        # self.goal['request_slots'][self.default_key] = 'UNK'
        # self.state['request_slots'][req_key] = 'UNK'

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])
        #print('Intial User Action:',user_response)

        return user_response

    def step(self, agent_action,stp):
        """
        Return the response of the user sim. to the agent by using rules that simulate a user.

        Given the agent action craft a response by using deterministic rules that simulate (to some extent) a user.
        Some parts of the rules are stochastic. Check if the agent has succeeded or lost or still going.

        Parameters:
            agent_action (dict): The agent action that the user sim. responds to

        Returns:
            dict: User sim. response
            int: Reward
            bool: Done flag
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        # Assertions -----
        # No UNK in agent action informs
        prev_slt_len = len(self.state['rest_slots'])
        #print('Prev Slot Length:',prev_slt_len)
        for value in agent_action['inform_slots'].values():
            assert value != 'UNK'
            if(agent_action['intent']!='Persuasion' or agent_action['intent']=='Repersuasion'):
                assert value != 'PLACEHOLDER'
        # No PLACEHOLDER in agent at all
        for value in agent_action['request_slots'].values():
            assert value != 'PLACEHOLDER'
        # ----------------

        self.state['inform_slots'].clear()
        self.state['intent'] = ''

        done = False
        success = NO_OUTCOME
        # First check round num, if equal to max then fail
        if agent_action['round'] == self.max_round:
            done = True
            success = FAIL
            self.state['intent'] = 'done'
            self.state['request_slots'].clear()
        else:
            agent_intent = agent_action['intent']
            if agent_intent == 'request':
                self._response_to_request(agent_action)
            elif agent_intent == 'inform':
                self._response_to_inform(agent_action)
            elif agent_intent == 'match_found':
                self._response_to_match_found(agent_action)
            elif agent_intent == 'done':
                self.state['request_slots'].clear()
                if(self.f):
                    success = 1
                else:
                    success = self._response_to_done(agent_action)
                self.state['intent'] = 'done'
                done = True
            elif(agent_intent == 'Sp'):            ###Agenr Specification Questions
                self._res_specifications(agent_action)
            elif agent_intent == 'Persuasion' or agent_intent == 'CredibilityAppeal' or agent_intent=='LogicalAppeal' or agent_intent=='EmotionalAppeal' or agent_intent=='PersonalAppeal' :
                self._response_to_Persuasion(agent_action)
            elif agent_intent == 'Repersuasion':
                self._response_to_Repersuasion(agent_action)
            elif agent_intent == 'UserGoalUpdate':
                self._response_to_UserGoalUpdate(agent_action)


        # Assumptions -------
        # If request intent, then make sure request slots
        if self.state['intent'] == 'request':
            assert self.state['request_slots']
        # If inform intent, then make sure inform slots and NO request slots
        if self.state['intent'] == 'inform':
            assert self.state['inform_slots']
            assert not self.state['request_slots']
        assert 'UNK' not in self.state['inform_slots'].values()
        assert 'PLACEHOLDER' not in self.state['request_slots'].values()
        # No overlap between rest and hist
        for key in self.state['rest_slots']:
            assert key not in self.state['history_slots']
        for key in self.state['history_slots']:
            assert key not in self.state['rest_slots']
        # All slots in both rest and hist should contain the slots for goal
        for inf_key in self.goal['inform_slots']:
            assert self.state['history_slots'].get(inf_key, False) or self.state['rest_slots'].get(inf_key, False)
        for req_key in self.goal['request_slots']:
            assert self.state['history_slots'].get(req_key, False) or self.state['rest_slots'].get(req_key,False), req_key
        # Anything in the rest should be in the goal
        for key in self.state['rest_slots']:
            assert self.goal['inform_slots'].get(key, False) or self.goal['request_slots'].get(key, False)
        #assert self.state['intent'] != ''
        # -----------------------

        user_response = {}
        user_response['intent'] = self.state['intent']
        user_response['request_slots'] = copy.deepcopy(self.state['request_slots'])
        user_response['inform_slots'] = copy.deepcopy(self.state['inform_slots'])
        #print('Current Rest Slots:',self.state['rest_slots'])
        reward = reward_function(success, self.max_round,prev_slt_len,len(self.state['rest_slots']),stp)




        return user_response,self.sent, reward, done, True if success is 1 else False






    def _response_to_request(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of request.

        There are 4 main cases for responding.

        Parameters:
            agent_action (dict): Intent of request with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_request_key = list(agent_action['request_slots'].keys())[0]
        # First Case: if agent requests for something that is in the user sims goal inform slots, then inform it
        if agent_request_key in self.goal['inform_slots']:
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
            self.state['request_slots'].clear()
            self.state['rest_slots'].pop(agent_request_key, None)
            self.state['history_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
        # Second Case: if the agent requests for something in user sims goal request slots and it has already been
        # informed, then inform it
        elif agent_request_key in self.goal['inform_slots'] and agent_request_key in self.state['history_slots']:
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = self.goal['inform_slots'][agent_request_key]
            self.state['request_slots'].clear()
            #assert agent_request_key not in self.state['rest_slots']
        # Third Case: if the agent requests for something in the user sims goal request slots and it HASN'T been
        # informed, then request it with a random inform
        elif agent_request_key in self.goal['request_slots'] and agent_request_key in self.state['rest_slots']:
            self.state['request_slots'].clear()
            self.state['intent'] = 'request'
            self.state['request_slots'][agent_request_key] = 'UNK'
            rest_informs = {}
            for key, value in list(self.state['rest_slots'].items()):
                if value != 'UNK':
                    rest_informs[key] = value
            if rest_informs:
                key_choice, value_choice = random.choice(list(rest_informs.items()))
                self.state['inform_slots'][key_choice] = value_choice
                self.state['rest_slots'].pop(key_choice)
                self.state['history_slots'][key_choice] = value_choice
        # Fourth and Final Case: otherwise the user sim does not care about the slot being requested, then inform
        # 'anything' as the value of the requested slot
        else:
            assert agent_request_key not in self.state['rest_slots']
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_request_key] = 'anything'
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_request_key] = 'anything'



    def _res_specifications(self,agent_action):
        agent_request_key = list(agent_action['request_slots'].keys())[0]
        val = 'UNK'
        if agent_request_key == 'Specifications':                    ###Specification
            print(self.state['inform_slots'])
            if(len(self.info)):
                key, val = random.choice(list((self.info).items()))
                self.state['intent'] = 'inform'
                self.state['inform_slots'][key] = val

                self.state['history_slots'][key] = val
                if(key in self.state['rest_slots']):
                    self.state['rest_slots'].pop(key)

                (self.info).pop(key)
                self.state['request_slots'].clear()
            else:
                self.state['intent'] = 'inform'
                self.state['inform_slots']['Specifications'] = 'Y'
                self.state['request_slots'].clear()

        elif agent_request_key == 'SpDone':
            #print('entered')                             ###Specification done ?
            if(len(self.info)<1):
                #print(self.state['inform_slots'])
                #print(self.state['rest_slots'])
                #print('No slot left')
                self.state['intent'] = 'inform'
                self.state['inform_slots']['SpDone'] = 'Y'
                self.state['request_slots'].clear()
            else:
                #print('Slot left')
                #print(self.state['rest_slots'])
                key, val = random.choice(list((self.info).items()))
                self.state['intent'] = 'inform'
                self.state['inform_slots'][key] = val
                if(key in self.state['rest_slots']):
                    self.state['rest_slots'].pop(key)
                self.state['history_slots'][key] = val
                self.state['request_slots'].clear()
                (self.info).pop(key)

            return




    def _response_to_Persuasion(self,agent_action):
        self.perno = self.perno + 1
        r = random.random()
        print('random number in persuasion function:',r)
        if(r<0.7 and len(self.state['rest_slots'])<=1 and self.p==2):
        #if(r<0.6 and self.p==2):

            if(self.strategy[self.ps]!='Persuasion' and agent_action['intent']==self.strategy[self.ps]):
                if len(self.chh):
                    self.chh.pop(self.amk,None)
                self.state['intent']='thanks'
                self.state['inform_slots'].clear()
                self.state['request_slots'].clear()
                self.constraint_check = SUCCESS
                self.state['intent'] = 'done'
                self.f = 1
            elif(self.strategy[self.ps]=='Persuasion' and agent_action['intent']==self.strategy[self.ps]):
                if(len(self.PMS)):
                    if agent_action['inform_slots'] in self.PMS:
                        if am_key in list(self.chh.keys()):
                            self.chh.pop(am_key,None)
                        self.state['intent']='thanks'
                        self.state['inform_slots'].clear()
                        self.state['request_slots'].clear()
                        self.constraint_check = SUCCESS
                        self.state['intent'] = 'done'
                        self.f = 1
                    else:
                        if(self.reject<2):
                            self.reject = self.reject+1
                            self.state['intent'] = 'reject'
                        else:
                            self.state['intent']='done'
                            self.state['inform_slots'].clear()
                            self.state['request_slots'].clear()
                            self.constraint_check = FAIL

                elif(r<0.4 and agent_action['inform_slots'] in self.GT):
                    if am_key in list(self.chh.keys()):
                        self.chh.pop(am_key,None)
                    self.state['intent']='thanks'
                    self.state['inform_slots'].clear()
                    self.state['request_slots'].clear()
                    self.constraint_check = SUCCESS
                    self.state['intent'] = 'done'
                    self.f = 1
                else:
                    if(self.reject<2):
                        self.reject = self.reject+1
                        self.state['intent'] = 'reject'
                    else:
                        self.state['intent']='done'
                        self.state['inform_slots'].clear()
                        self.state['request_slots'].clear()
                        self.constraint_check = FAIL


            else:
                if(self.reject<2):
                    self.reject = self.reject+1
                    self.state['intent'] = 'reject'
                else:
                    self.state['intent']='done'
                    self.state['inform_slots'].clear()
                    self.state['request_slots'].clear()
                    self.constraint_check = FAIL

        elif(r<0.6 and len(self.state['rest_slots'])>1 and self.p==2):

            if(agent_action['intent']==self.strategy[self.ps]):
                if(len(self.chh)):
                    self.chh.pop(self.amk,None)
            if(len(self.info)):
                key,val = random.choice(list(self.info.items()))
                self.info.pop(key,None)
                self.state['intent'] = 'inform'
                self.state['inform_slots'][key] = val
                self.state['request_slots'].clear()
                self.state['history_slots'][key] = val
                if(key in self.state['rest_slots']):
                    self.state['rest_slots'].pop(key,None)
            elif(len(self.state['rest_slots'])>1):
                key,val = random.choice(list(self.state['rest_slots'].items()))

                if(val=='UNK'):
                    self.state['intent']='request'
                    self.state['request_slots'][key] = 'UNK'
                    if(key in self.state['rest_slots'] and key!=self.default_key):
                        self.state['rest_slots'].pop(key,None)
                    self.state['inform_slots'].clear()
                    if(key!=self.default_key):
                        self.state['history_slots'][key] = 'UNK'



        else:
            if(self.reject<2):
                self.reject = self.reject+1
                self.state['intent'] = 'reject'
            else:
                self.state['intent']='done'
                self.state['inform_slots'].clear()
                self.state['request_slots'].clear()
                self.constraint_check = FAIL


    def _response_to_Repersuasion(self,agent_action):
        r = random.random()
        if(r<0.6 and len(self.state['rest_slots'])<=1 and self.p ==2):
            self.state['intent']='thanks'
            self.state['inform_slots'].clear()
            self.state['request_slots'].clear()
            self.constraint_check = SUCCESS
            self.state['intent'] = 'done'
            self.f = 1
            # agent_informs = agent_action['inform_slots']
            # self.state['rest_slots'].pop(self.default_key, None)
            # self.state['history_slots'][self.default_key] = str(agent_informs[self.default_key])
            # self.state['request_slots'].pop(self.default_key, None)

        elif(r<0.6 and len(self.state['rest_slots'])>1):
            if(len(self.info)):
                key,val = random.choice(list(self.info.items()))
                self.info.pop(key,None)
                self.state['intent'] = 'inform'
                self.state['inform_slots'][key] = val
                self.state['request_slots'].clear()
                self.state['history_slots'][key] = val
                if(key in self.state['rest_slots']):
                    self.state['rest_slots'].pop(key,None)
            elif(len(self.state['rest_slots'])>1):
                key,val = random.choice(list(self.state['rest_slots'].items()))
                if(val=='UNK'):
                    self.state['intent']='request'
                    self.state['request_slots'][key] = 'UNK'
                    if(key in self.state['rest_slots'] and key!=self.default_key):
                        self.state['rest_slots'].pop(key,None)
                    self.state['inform_slots'].clear()
                    if(key!=self.default_key):
                        self.state['history_slots'][key] = 'UNK'


        else:
            if(self.reject<3):
                self.reject = self.reject+ 1

                self.state['intent'] = 'reject'
            else:
                self.state['intent']='done'
                self.state['inform_slots'].clear()
                self.state['request_slots'].clear()
                self.constraint_check = FAIL

    def _response_to_UserGoalUpdate(self,agent_action):


        if(len(self.state['rest_slots'])>1):
            self.state['intent'] = 'reject'
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()

        else:
            if(self.p==2 and len(self.am) and self.perno):
                self.state['intent']='inform'
                self.state['request_slots'].clear()

                key_choice, value_choice = random.choice(list(self.am.items()))
                self.state['inform_slots'][key_choice] = value_choice
                self.am.pop(key_choice,None)
            else:
                self.state['intent']='reject'
                self.state['inform_slots'].clear()
                self.state['request_slots'].clear()







    def _response_to_inform(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of inform.

        There are 2 main cases for responding. Add the agent inform slots to history slots,
        and remove the agent inform slots from the rest and request slots.

        Parameters:
            agent_action (dict): Intent of inform with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """

        agent_inform_key = list(agent_action['inform_slots'].keys())[0]
        agent_inform_value = agent_action['inform_slots'][agent_inform_key]

        assert agent_inform_key != self.default_key

        # Add all informs (by agent too) to hist slots
        self.state['history_slots'][agent_inform_key] = agent_inform_value
        # Remove from rest slots if in it
        self.state['rest_slots'].pop(agent_inform_key, None)
        # Remove from request slots if in it
        self.state['request_slots'].pop(agent_inform_key, None)

        # First Case: If agent informs something that is in goal informs and the value it informed doesnt match,
        # then inform the correct value
        if agent_inform_value != self.goal['inform_slots'].get(agent_inform_key, agent_inform_value):
            self.state['intent'] = 'inform'
            self.state['inform_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
            self.state['request_slots'].clear()
            self.state['history_slots'][agent_inform_key] = self.goal['inform_slots'][agent_inform_key]
        # Second Case: Otherwise pick a random action to take
        else:
            # - If anything in state requests then request it
            if self.state['request_slots']:
                self.state['intent'] = 'request'
            # - Else if something to say in rest slots, pick something
            elif self.state['rest_slots']:
                def_in = self.state['rest_slots'].pop(self.default_key, False)
                if self.state['rest_slots']:
                    key, value = random.choice(list(self.state['rest_slots'].items()))
                    if value != 'UNK':
                        self.state['intent'] = 'inform'
                        self.state['inform_slots'][key] = value
                        self.state['rest_slots'].pop(key)
                        self.state['history_slots'][key] = value
                    else:
                        self.state['intent'] = 'request'
                        self.state['request_slots'][key] = 'UNK'
                else:
                    self.state['intent'] = 'request'
                    self.state['request_slots'][self.default_key] = 'UNK'
                if def_in == 'UNK':
                    self.state['rest_slots'][self.default_key] = 'UNK'
            # - Otherwise respond with 'nothing to say' intent
            else:
                self.state['intent'] = 'thanks'

    def _response_to_match_found(self, agent_action):
        """
        Augments the state in response to the agent action having an intent of match_found.

        Check if there is a match in the agent action that works with the current goal.

        Parameters:
            agent_action (dict): Intent of match_found with standard action format (including 'speaker': 'Agent' and
                                 'round_num': int)
        """


        if(len(self.state['rest_slots'])<=1 and len(self.am)<1):
            agent_informs = agent_action['inform_slots']
            self.state['intent'] = 'thanks'
            self.constraint_check = SUCCESS
            assert self.default_key in agent_informs
            ##Modified
            #if agent_informs[self.default_key] == 'no match available':
                #self.constraint_check = FAIL
            ##Modified end
            # Check to see if all goal informs are in the agent informs, and that the values match

            #
            #for key, value in self.goal['inform_slots'].items():


            for key, value in self.chh.items():
                assert value != None
                # For items that cannot be in the queries don't check to see if they are in the agent informs here
                if key in self.no_query:
                    continue
                # Will return true if key not in agent informs OR if value does not match value of agent informs[key]
                if value != agent_informs.get(key, None):
                    self.constraint_check = FAIL

                    break
        elif(len(self.state['rest_slots'])<=1 and len(self.am)>0 and self.p ==1):
            key,val = random.choice(list(self.am.items()))
            self.state['intent']='inform'
            self.state['inform_slots'][key]=val
            self.state['request_slots'].clear()
            self.am.pop(key,None)
            return
        else:
            self.state['intent']='reject'
            self.state['request_slots'].clear()
            self.state['inform_slots'].clear()



        if self.constraint_check == FAIL:
            self.state['intent'] = 'reject'
            self.state['request_slots'].clear()

        if self.constraint_check ==SUCCESS:
            agent_informs = agent_action['inform_slots']
            self.state['rest_slots'].pop(self.default_key, None)
            self.state['history_slots'][self.default_key] = str(agent_informs[self.default_key])
            self.state['request_slots'].pop(self.default_key, None)

    def _response_to_done(self,agent_action):
        """
        Augments the state in response to the agent action having an intent of done.

        If the constraint_check is SUCCESS and both the rest and request slots of the state are empty for the agent
        to succeed in this episode/conversation.

        Returns:
            int: Success: -1, 0 or 1 for loss, neither win nor loss, win
        """

        if self.constraint_check == FAIL:
            return FAIL

        if not self.state['rest_slots']:
            assert not self.state['request_slots']
        if self.state['rest_slots']:
            return FAIL

        # TEMP: ----
        if(self.state['history_slots'][self.default_key] == 'no match available'):
            self.constraint_check == FAIL
            return FAIL
        else:


            assert self.state['history_slots'][self.default_key] != 'no match available'

            match = copy.deepcopy(self.database[int(self.state['history_slots'][self.default_key])])




            #for key, value in self.goal['inform_slots'].items():
            for key, value in self.chh.items():
                if value =='anything':
                    self.chh.pop(key,None)
                assert value != None
                if key in self.no_query:
                    continue
                if value != match.get(key, None):
                    if (True is False, 'match: {}\ngoal: {}'.format(match, self.chh)):
                        break
                    else:
                        return FAIL

            return SUCCESS
