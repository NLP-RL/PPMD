from nus import UserSimulator
from error_model_controller import ErrorModelController
from test_agent import DQNAgent
from state_tracker import StateTracker
import pickle, argparse, json
from User_Terminal import User
from utils import remove_empty_slots
from utils import Sent_Score
from dialogue_config import FAIL, SUCCESS, usersim_intents, all_slots
from utils import reward_function
import pickle
import random
import os
import re
import pandas as pd
from Template_NLG import NLG
from BERT_NLU import TagsVectorizer
from BERT_NLU import predict2
import json
from Template_NLG import NLG

import User_Terminal

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"
u_r =0
a_r = 0
Agent_Actions = []
User_Actions = []
done = False
a_q = 0
u_q = 0
p = 0
xx = 0
ui = 0
pua = {'intent':'','request_slots':{},'inform_slots':{}}


from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
app = Flask(__name__)
CORS(app)


@app.route('/dialogue', methods=['GET', 'POST'])
def dialogue():
    #-----Ashok added---------
    global u_r
    global a_r
    global Agent_Actions
    global User_Actions
    global done
    global a_q
    global u_q
    global p
    global xx
    global ui
    global pua
    global agent_action
    global strategy
    content = request.json
    print('printing content'+ str(content))
    input_sentence = content['sentence']
    username = content['username']


    reset_constant = content['k']
    strategy =  {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'LogicalAppeal',4:'PersonalAppeal',5:'Persuasion'}
    if int(reset_constant) == 1:
        xx = 0
    print(input_sentence)
    print('printin XX values', xx)
    if xx==0:
        string,url =  reset(input_sentence,username)
        xx = xx+1
        return jsonify({"output" : str(string),"url" : str(url)})
        #added by ashok


    #state,stateinfo = state_tracker.get_state()
    if not done:

        xx = xx + 1

        user_action,ps,sent, reward, done, success = user.step(agent_action,xx,input_sentence)
        print('user action',user_action)
        state_tracker.psupdate(ps)
        print('Current Predicted Personality: ',strategy[ps])
        state_tracker.update_state_user(user_action,reward,sent)
        state,stateinfo = state_tracker.get_state()
        agent_action_index, agent_action = dqn_agent.get_action(state)
        print('state in dailouge function',state)
        print('agent action in dailouge fun',agent_action)
        #user_action,sent, reward, done, success = user.step(agent_action,xx,input_sentence)
        if user_action not in User_Actions:
            User_Actions = User_Actions + [user_action]
        else:
            u_r  = 1
        if(user_action['intent']=='request'):
            u_q = 1
        if(user_action['intent']=='inform' or user_action['intent']=='feedback' or ui == 'Phn_Booking' or user_action['intent']=='preq'):
            ui = 1
        else:
            ui = 0
        if xx==1:
            U = {'intent':'','request_slots':{},'inform_slots':{}} ##dummy
        else:
            U = User_Actions[-2]
        print('PUA: ',U)
        reward = 0
        pen = 0 ##dummy variable
        ss,prob,pc = Sent_Score(state,a_r,u_r,pen,sent,pua,agent_action,stateinfo)
        print('pc in main function: ',pc)
        #state_tracker.update_state_user(user_action,reward,sent)
        pua = user_action
        state,stateinfo = state_tracker.get_state(done)
        #agent_action_index, agent_action = dqn_agent.get_action(state)

        v,available_result,url =  state_tracker.update_state_agent(agent_action,ui)
        print('len of Available result:', len(available_result))
        print("printing done ####################################",str(done))
        print("Printing success variable###########################",str(success))
        agent_text_response = nlg.act_response(agent_action,0,v,available_result,url,success)
        try:
            url = re.search("(?P<url>http?://[^\s]+)", agent_text_response).group("url")
            string = re.sub("(?P<url>http?://[^\s]+)",'', agent_text_response)
        except Exception as err:
            print(err)
            url = ''
            string = agent_text_response

    else:
        print('Dialogue Completed !!!')
        xx = 0
        return jsonify({"output" : 'Dialogue Completed !!!', "url" : ''})
    return jsonify({"output" : str(string),"url" : str(url)})

@app.route('/reset', methods=['GET', 'POST'])
def reset(input_sentence,username):

    global u_r
    global a_r
    global Agent_Actions
    global User_Actions
    global done
    global a_q
    global u_q
    global p
    global xx
    global ui
    #global success
    User_Terminal.success = 0
    ui = 0
    xx = 0     ##Dialogue Turns

    #content = request.json
    #print('printing content'+ str(content))
    #input_sentence = content['sentence']
    print(input_sentence)
    # i = random.randint(0,(len(persona)-1))

    # P = [persona['Color'][i],persona['Photographer'][i],persona['Brand'][i],persona['AgeRange'][i],persona['OS'][i]]
    # if P[3]==0:
    #     yr = '2017'
    # elif P[3]==1:
    #     yr = '2016'
    # elif P[3]==2:
    #     yr = '2015'
    # PP = {'Color':P[0],'Brand':P[2],'RY':yr,'OS':P[4]}
    # state_tracker.reset(PP)
    #i = random.randint(0,(len(persona)-1))
    if username=='ashok':
        i=0
    elif username == 'sumit':
        i=1
    elif username == 'abhisek':
        i=2
    elif username == 'shubhashis':
        i=3
    elif username == 'roshni':
        i=4
    else:
        i = random.randint(0,(len(persona)-1))

    #tot_slt,user_action = user.reset(PP)


    P = [persona['FavColor'][i],persona['Photographer '][i],persona['FavBrand'][i],persona['FavOS'][i],persona['Gender'][i]]

    PP = {'Color':P[0],'Brand':P[2],'P_Camera':P[1],'OS':P[3],'Gender':P[4]}
    print('Persona: ',PP)


    # Then pick an init user action
    #j = random.randint(0,4)
    tot_slt,user_action,ps = user.reset(input_sentence)
    state_tracker.reset(PP,ps)


    if(user_action['intent']=='inform' or user_action['intent']=='feedback' or user_action['intent']=='Phn_Booking' or user_action['intent'] == 'preq'):
        ui = 1
    else:
        ui = 0
    #tot_slt,user_action = user.reset(input_sentence)
    #text_response = nlg.act_response(user_action,1,-1)
    if(user_action['intent']=='inform' or user_action['intent']=='feedback' or user_action['intent']=='Phn_Booking' or user_action['intent'] == 'preq'):
        ui = 1
    else:
        ui = 0

    state_tracker.update_state_user(user_action,0,0)
    dqn_agent.reset()
    u_r =0
    a_r = 0
    Agent_Actions = []
    User_Actions = []
    done = False
    a_q = 0
    u_q = 0
    p = 0



    state,stateinfo = state_tracker.get_state()

    xx = xx + 1
    global agent_action
    agent_action_index, agent_action = dqn_agent.get_action(state)

    print('state in rest function',state)
    print('agent action in reset function',agent_action)
    if(agent_action['intent']=='request' and u_q == 1):
        p = 1
    v,available_result,url =  state_tracker.update_state_agent(agent_action,ui)
    agent_text_response = nlg.act_response(agent_action,0,v,available_result,url,User_Terminal.success)
    if(agent_action_index in Agent_Actions):
            a_r = 1
    else:
        Agent_Actions = Agent_Actions + [agent_action_index]

    User_Actions = User_Actions + [user_action]

    ps = nlg.pcReturn()

    try:
        url = re.search("(?P<url>http?://[^\s]+)", agent_text_response).group("url")
        string = re.sub("(?P<url>http?://[^\s]+)",'', agent_text_response)
    except Exception as err:
        print(err)
        url = ''
        string = agent_text_response

    return string,url,ps



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--constants_path', dest='constants_path', type=str, default='')
    args = parser.parse_args()
    persona = pd.read_csv('Data/up_Persona.csv')
    params = vars(args)

    CONSTANTS_FILE_PATH = 'configs.json'
    if len(params['constants_path']) > 0:
        constants_file = params['constants_path']
    else:
        constants_file = CONSTANTS_FILE_PATH

    with open(constants_file) as f:
        constants = json.load(f)

    file_path_dict = constants['db_file_paths']
    DATABASE_FILE_PATH = file_path_dict['database']
    DICT_FILE_PATH = file_path_dict['dict']
    USER_GOALS_FILE_PATH = file_path_dict['user_goals']

    run_dict = constants['run']
    USE_USERSIM = run_dict['usersim']
    nlg = NLG()
    NUM_EP_TEST = run_dict['num_ep_run']
    MAX_ROUND_NUM = run_dict['max_round_num']

    database = pickle.load(open(DATABASE_FILE_PATH, 'rb'), encoding='latin1')

    remove_empty_slots(database)

    db_dict = pickle.load(open(DICT_FILE_PATH, 'rb'), encoding='latin1')

    user_goals = pickle.load(open(USER_GOALS_FILE_PATH, 'rb'), encoding='latin1')

    if USE_USERSIM:
        user = UserSimulator(user_goals, constants, database)
    else:
        user = User(constants)
    state_tracker = StateTracker(database, constants)
    dqn_agent = DQNAgent(state_tracker.get_state_size(), constants)
    #test_run()
    #app.run(host='0.0.0.0')
    app.run(debug = False, host='172.16.26.58',port=5000)
    #app.run(port='5000')
