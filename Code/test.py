from nus import UserSimulator
from error_model_controller import ErrorModelController
from dqn_test import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json, math
from utils import remove_empty_slots
#from user import User
from utils import Sent_Score,PMeR
import os
import random
import pandas as pd
from utils import graph
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

if __name__ == "__main__":
    # Can provide constants file path in args OR run it as is and change 'CONSTANTS_FILE_PATH' below
    # 1) In terminal: python train.py --constants_path "constants.json"
    # 2) Run this file as is
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    params = vars(args)


    # Load constants json into dict
    CONSTANTS_FILE_PATH = 'configs.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    # Load file path constants
    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    persona = pd.read_csv('Data/up_Persona.csv')
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['test_user_goals']

    ##

    # Load run constants
    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    WARMUP_MEM = run_dict['warmup_mem']
    NUM_EP_TRAIN = run_dict['num_ep_run']
    TRAIN_FREQ = run_dict['train_freq']
    MAX_ROUND_NUM = run_dict['max_round_num']
    SUCCESS_RATE_THRESHOLD = run_dict['success_rate_threshold']

    # Load movie DB
    # Note: If you get an unpickling error here then run 'pickle_converter.py' and it should fix it
    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    # Clean DB
    remove_empty_slots(database)

    # Load movie dict
    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    # Load goal File
    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    # Init. Objects
    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    emc = ErrorModelController(db_dict, constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)
    pmer = 0



def run_round(state,Agent_Actions,User_Actions,stp,stateinfo,U ,warmup=False):
    global pmer
    u_r =0                 ##User Repeatition
    a_r = 0                ##Agent Repeatition
    a_q = 0                ##Agent Question
    pen = 0         ##User asked Question Agent replied Question
    # 1) Agent takes action given state tracker's representation of dialogue (state)
    agent_action_index, agent_action = dqn_agent.get_action(state, use_rule=warmup)
    print('Agent Action:',agent_action)
    if(agent_action['intent']=='request'):
        a_q = 1
    if(agent_action_index in Agent_Actions):
        a_r = 1
    else:
        Agent_Actions = Agent_Actions + [agent_action_index]
    if(U['intent'] == 'request' and a_q ==1):
        pen = 1
    # 2) Update state tracker with the agent's action
    state_tracker.update_state_agent(agent_action,1)
    # 3) User takes action given agent action
    user_action,sent, reward, done, success = user.step(agent_action,stp)
    print('User_Action:',user_action)
    print('Task Oriented Reward: ',reward)


    temp = {'intent':'','inform_slots':'','request_slots':''}
    temp['intent'] = user_action['intent']
    temp['inform_slots'] = user_action['inform_slots']
    temp['request_slots'] = user_action['request_slots']

    #print('User_Actions:',User_Actions)
    #print('Temp:',temp)
    #print('User_Actions:',User_Actions)
    if temp in User_Actions:
        u_r = 1
        print('!!!User Repeatition')
    else:
        User_Actions = User_Actions + [temp]

    #if not done:
        # 4) Infuse error into semantic frame level of user action
        #emc.infuse_error(user_action)

    ss,prob,pc = Sent_Score(state,a_r,u_r,pen,sent,U,agent_action,stateinfo)
    pmer = PMeR(state,a_r,u_r,pen,sent,U,agent_action,stateinfo,reward, ss, pc, done,success)
    reward = reward + ss
    # 5) Update state tracker with user action
    state_tracker.update_state_user(user_action,reward,sent)
    # 6) Get next state and add experience
    next_state,stateinfo = state_tracker.get_state(done)
    dqn_agent.add_experience(state, agent_action_index, reward, next_state, done)
    print('Total Reward:',reward)
    return next_state, reward, done, success,Agent_Actions,User_Actions,stateinfo,temp


def warmup_run():
    """
    Runs the warmup stage of training which is used to fill the agents memory.

    The agent uses it's rule-based policy to make actions. The agent's memory is filled as this runs.
    Loop terminates when the size of the memory is equal to WARMUP_MEM or when the memory buffer is full.

    """

    print('Warmup Started...')
    total_step = 0
    s = 0
    while total_step != WARMUP_MEM and not dqn_agent.is_memory_full():
        q = 0
        #print('WarmupPhase:',total_step)
        # Reset episode
        ua= episode_reset()
        Agent_Actions = []
        User_Actions = []
        User_Actions = User_Actions + [ua]
        done = False
        # Get initial state from state tracker
        state,stateinfo = state_tracker.get_state()
        l = 0
        while not done:
            l = l+1
            next_state, R, done, Succ,Agent_Actions,UA, q,stateinfo_n = run_round(state,Agent_Actions,User_Actions,l,q,stateinfo,warmup=True)
            stateinfo = stateinfo_n
            User_Actions = UA
            total_step += 1
            state = next_state
        if(Succ==1):
            s = s+1
    print('Total Success:',s)
    print('...Warmup Ended')


def train_run():
    """
    Runs the loop that trains the agent.

    Trains the agent on the goal-oriented chatbot task. Training of the agent's neural network occurs every episode that
    TRAIN_FREQ is a multiple of. Terminates when the episode reaches NUM_EP_TRAIN.

    """

    print('Training Started...')
    global pmer
    episode = 0
    period_reward_total = 0
    period_success_total = 0
    success_rate_best = 0.0
    succ_rate = []
    avg_rewd = []
    l = 0
    L = []
    x = 0
    r=0
    N = []
    PMR = []
    while episode < NUM_EP_TRAIN:
        Agent_A = []
        prev_user_action = {}
        User_A = []
        qq = 0
        print('@@@Episode No:',episode)
        ua = episode_reset()
        User_A = User_A + [ua]
        episode += 1
        done = False
        state,stateinfo = state_tracker.get_state()
        ppmer = 0
        while not done:
            x = x + 1
            l = l + 1
            next_state, reward, done, success,Agent_Actions,UA,stateinfo_n,UU = run_round(state,Agent_A,User_A,x,stateinfo,ua)
            ua = UU
            stateinfo = stateinfo_n
            Agent_A = Agent_Actions
            User_A = UA
            period_reward_total += reward
            r = r + reward
            ppmer = ppmer + pmer
            state = next_state
        z = ppmer/x
        print('DL:',x)
        print('z:',z)
        if z>1:
            print(1/0)
        PMR = PMR + [ppmer/x]
        ppmer = 0
        pmer = 0



        print('Taken Length:',x)
        x = 0
        print('Cummulative Reward:',r)
        r = 0
        period_success_total += success

        # Train
        if episode % TRAIN_FREQ == 0:
            #print('Episode No:',episode)
            avg_len = l/TRAIN_FREQ

            #print('Episodic Avg Length:',avg_len)
            L = L + [avg_len]
            N = N + [period_success_total]
            # Check success success_rate_threshold
            success_rate = period_success_total / TRAIN_FREQ
            succ_rate = succ_rate + [success_rate]
            avg_reward = period_reward_total / TRAIN_FREQ
            #print('Episodic Reward:',period_reward_total)
            avg_rewd = avg_rewd + [avg_reward]

            # Flush
            print('Episode:',episode)
            print('Avg Reward:',avg_reward)
            print('Avg length:',avg_len)
            # if success_rate >= success_rate_best and success_rate >= SUCCESS_RATE_THRESHOLD:
            #     dqn_agent.empty_memory()
            # Update current best success rate
            # if success_rate > success_rate_best:
            #     print('Episode: {} NEW BEST SUCCESS RATE: {} Avg Reward: {}' .format(episode, success_rate, avg_reward))
            #     success_rate_best = success_rate
            #     dqn_agent.save_weights()
            period_success_total = 0
            period_reward_total = 0
            l = 0
            # Copy
            # dqn_agent.copy()
            # Train
            # dqn_agent.train()
    print('***********Training Result***********')
    print('Success Rate:',succ_rate)
    print('Avg Reward: ',avg_rewd)
    print('Avg Length:',L)
    print('Toatal Episode:',len(succ_rate))
    print('PMeR : ',PMR)
    print('Result of last 30 episodes')
    s_r = succ_rate[-30:]
    rewd = avg_rewd[-30:]
    ll = L[-30:]
    ppmr = PMR[-30:]
    print('Success Rate : ',sum(s_r)/len(s_r))
    print('Avg Episodic Reward:',sum(rewd)/len(rewd))
    print('Dailogue Length:',sum(ll)/len(ll))
    print('pmer : ',sum(ppmr)/len(ppmr))
    result = {'SR':succ_rate,'DL':L,'R':avg_rewd,'PMR':PMR}


    # graph(len(avg_rewd),avg_rewd,'Avg Reward',filename=constants['agent']['rew_img'])
    # graph(len(L), L, 'Length',filename=constants['agent']['len_img'])
    # graph(len(succ_rate),succ_rate,'Success rate',filename=constants['agent']['suc_img'])


def episode_reset():
    """
    Resets the episode/conversation in the warmup and training loops.

    Called in warmup and train to reset the state tracker, user and agent. Also get's the initial user action.

    """

    # First reset the state tracker
    global pmer
    i = random.randint(0,(len(persona)-1))
    P = [persona['FavColor'][i],persona['Photographer '][i],persona['FavBrand'][i],persona['FavOS'][i],persona['Gender'][i]]

    PP = {'Color':P[0],'Brand':P[2],'P_Camera':P[1],'OS':P[3],'Gender':P[4]}
    print('Persona: ',PP)

    # Then pick an init user action
    j = random.randint(0,4)
    tot_slt,user_action,ps = user.reset(PP,j)
    state_tracker.reset(PP,ps)
    print('intial User Action::',user_action)
    # Infuse with error
    emc.infuse_error(user_action)
    pmer = 0
    # And update state tracker
    state_tracker.update_state_user(user_action,0,0)
    # Finally, reset agent
    dqn_agent.reset()
    rn = user_action.pop('round')
    sp = user_action.pop('speaker')
    return user_action


#warmup_run()
train_run()
