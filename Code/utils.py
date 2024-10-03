from dialogue_config import FAIL, SUCCESS,all_slots
import matplotlib.pyplot as pyplot
from Template_NLG import NLG

import numpy as np

nlg = NLG()

def convert_list_to_dict(lst):
    """
    Convert list to dict where the keys are the list elements, and the values are the indices of the elements in the list.

    Parameters:
        lst (list)

    Returns:
        dict
    """

    if len(lst) > len(set(lst)):
        raise ValueError('List must be unique!')
    return {k: v for v, k in enumerate(lst)}


def remove_empty_slots(dic):
    """
    Removes all items with values of '' (ie values of empty string).

    Parameters:
        dic (dict)
    """

    for id in list(dic.keys()):
        for key in list(dic[id].keys()):
            if dic[id][key] == '':
                dic[id].pop(key)


def reward_function(success, max_round,prev_slt_len,current_slt_len,stp):
    """
    """



    if success == FAIL:
        reward = -3 * max_round
        print('&Failed&')
    elif success == SUCCESS:
        print('*****Success***')
        reward = 5 * (max_round - stp)
    elif((prev_slt_len-current_slt_len)>0):      ##for Reducing REst SLot/ Increasing Slot
        reward = 2*(prev_slt_len-current_slt_len)
        print('Rewarded for Increasing Slots:',reward)
    else:
        reward = -1.2     ### For Each utterance
    print('Step:',stp)
    return reward

##Neutral and Negative (1.Agent Action Repeatition 2.User Action Repeatition )
def Sent_Score(state,a_r,u_r,p,sent,user_action,agent_action,stateinfo):
    ss = 0
    pc = 0

    n, gu, p,avslt,unavslt,PSM,Slots,Personalilty = stateinfo[0],stateinfo[1],stateinfo[2],stateinfo[3],stateinfo[4],stateinfo[5],stateinfo[6],stateinfo[7]
    print('Sent_Score')
    print('Ava slot',avslt)
    if(a_r==1):             ##Agent Action Repeatition
         ss = ss - 30
         print('Penalized for Agent Action Repeatition')
    if(u_r==1):             ##User Action Repeatition
        ss = ss -3
        print('Pemalized as Agent was not able to answer in First Go')
    if(p == 1):             ##Inappropriate Reply by Agent (req by user -> req by agent)
        ss = ss - 3
        print('Penalized for an Inappropriate Action')


    if(user_action['intent']=='preq'):
        if agent_action['intent']!='Sp':
            ss = ss - 2
    print('user_action:00',user_action)
    if(user_action['intent']=='request'):
        att_req = list(user_action['request_slots'].keys())[0]
        if agent_action['intent']=='inform':
            att_rep = list(agent_action['inform_slots'].keys())[0]
            if(att_req==att_rep):
                ss = ss + 3
            else:
                ss = ss - 5
        else:
            ss = ss - 5
            print('Penalized for not taking proper inform')
    else:
        if(agent_action['intent']=='inform'):
            ss = ss -5


    if(user_action['intent']=='Phn_Booking'):
        if agent_action['intent']=='match_found' :
            ss = ss + 5
            print('Rewarded')
    if(agent_action['intent']=='match_found'):
        ss = ss + 0.1
    if(user_action['intent']!='done' and agent_action['intent']=='done'):
        ss = ss -30

    print('Inside Reward Model Unavalabaility:',stateinfo[4])
    strategy = {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'LogicalAppeal',4:'PersonalAppeal',5:'Persuasion'}
    #strategy = {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'PersonalAppeal',4:'Persuasion'}
    strategy_lst = list(strategy.values())
    if(gu): ##GU
        print('Personality: ',strategy[Personalilty])
        print('Persuassion Stage:',p)
        print('Inside Reward Model, Unavslt:',unavslt)
        if(p<2):
            if(agent_action['intent'] in strategy_lst):
                if agent_action['intent'] == strategy[Personalilty]:
                    ss = ss + 10
                    pc = 1
                    if (agent_action['intent']=='Persuasion'):
                        k = list(agent_action['inform_slots'].keys())[0]

                        if(len(PSM)):
                            print('USER Persona:',PSM)
                            if k in PSM:
                                ss = ss + 7
                                print('Persona Aware action')
                            else:
                                ss = ss - 5
                                print('Punished Not Persona Aware')
                        else:

                            if k in avslt:
                                ss = ss + 7*(2-p)
                                print('Rewared for avalabaility Persuassion :')
                            elif len(avslt)==0:
                                if(k=='features'):
                                    temp = 7*(2-p)
                                    ss = ss + 7*(2-p)
                                    print('Rewared for avalabaility Persuassion :',temp)
                                else:
                                    ss = ss -4

                            else:
                                print('Punished for Improper Persuassion')
                                ss = ss - 17
                else:
                    ss = ss - 25
            else:
                ss = ss - 35
        elif(p>=1):
            if(p < 3 and agent_action['intent']=='Repersuasion'):
                ss = ss + 7 * (4-p)

            elif(p>=3 and agent_action['intent']=='UserGoalUpdate'):
                ss = ss + 10
            else:
                ss = ss - 5
    else:
        if(agent_action['intent'] in strategy_lst or agent_action['intent']=='Repersuasion' or agent_action['intent']=='UserGoalUpdate'):
            print('Punished for taking Persuassive action')
            ss = ss -25


    if(sent == 0):
        prob = (ss + 12)/18

    nlg.pcUpdate(pc)
    print('Sentiment Reward :',ss)


    return ss,prob,pc

def PMeR(state,a_r,u_r,p,sent,user_action,agent_action,stateinfo,reward,ss,pc, done,success):
    pmer = 0
    n, gu, p,avslt,unavslt,PSM,Slots,Personalilty = stateinfo[0],stateinfo[1],stateinfo[2],stateinfo[3],stateinfo[4],stateinfo[5],stateinfo[6],stateinfo[7]
    strategy = {0:'LogicalAppeal',1:'CredibilityAppeal',2:'EmotionalAppeal',3:'LogicalAppeal',4:'PersonalAppeal',5:'Persuasion'}
    print('Personality: ',strategy[Personalilty])
    print('Persuassion Stage:',p)
    print('Inside Reward Model, Unavslt:',unavslt)
    print('succc:', success)
    print(done)
    if(gu):
        if ss<0 : #Persona aware response
            pmer = pmer - 0.3
            print('R1')
        if pc:
            pmer = pmer + 0.3
            print('R2')
        if done and success:
            pmer = 1
            print('R3')
        if done and success==False:
            pmer = -1
    print('PMer: ', pmer)
    return pmer









###PMeR
#1. persona aware response, user sentiment and end goal









def graph(n, P, label, filename):
    pyplot.figure()
    ep= list(range(0,n))
    pyplot.plot(ep,P)
    pyplot.ylabel(label)
    pyplot.xlabel('Episode')
    pyplot.savefig(filename)
