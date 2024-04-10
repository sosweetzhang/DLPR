# coding: utf-8
import gym
import networkx as nx
import torch
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import sys



sys.path.append('../')
from KT import Agent_KT
from AC import ActorCritic,Data_P
from PPO import PPO,Data_L



def convert_knowledge_structure(knowledge_structure):  # env.knowledge_structure

    knowledge_structure_dict = {}
    know_keys = list(dict(knowledge_structure.succ).keys())
    for key in know_keys:
        knowledge_structure_dict[key] = list(dict(dict(knowledge_structure.succ)[key]).keys())
    return knowledge_structure_dict


def retrive_state(state,relation):
    know_state = list()
    for know in relation.keys():
        know_state.append(torch.gather(state.squeeze(0), 0, torch.tensor(relation[know])).mean())

    know_state = torch.tensor(know_state)

    return know_state.unsqueeze(0)



def train(env, L_agent, P_agent, max_episode_num, batch_size):

    L_max_steps = 10
    L_rewards = []
    all_logs_episode_L = []
    all_logs_episode_P = []

    for episode in tqdm.tqdm(range(max_episode_num),desc="Apisode"):

        env.reset()
        logs_episode_L = []
        logs_episode_P = []

        init_profile, _ = env.begin_episode()

        Know_G = nx.DiGraph()
        Know_G.add_edges_from(list(env.knowledge_structure.edges))

        target = list(init_profile["target"])

        logs_episode_L.append({"learning targets":target})
        init_logs = init_profile["logs"]

        init_ques = []
        init_ans = []
        for log in init_logs:
            init_ques.append(int(log[0]))
            init_ans.append(int(log[1]))

        last_tor = 6 # demo
        last_prac_num = 5 # demo

        logs_episode_L.append({"length of initial logs":len(init_ques)})

        KT = Agent_KT()
        state = KT.forward_state(init_ques,init_ans)
        P_state = state[:, -1, :]
        next_P_state = P_state

        L_state = retrive_state(P_state, env.know_item)
        logs_episode_L.append({"init knowledge state":L_state})

        l_steps = 0
        l_knows = []
        know = 0

        while True:
            l_steps += 1
            know, tolerance, init_diff = L_agent.take_action(L_state,know,target,Know_G,k_hop=1,threshhold=0.6,last_tor=last_tor,last_prac_num=last_prac_num) # todo: L-agent select a kc, give tolerance and init diff to P-agent

            l_knows.append(know)
            p_steps = 0
            p_step_reward = 0

            logs_know_P = []
            logs_know_P.append("learning step {} study {} tolerance is {} and init_diff is {}".format(l_steps,know,tolerance,init_diff))

            for i in tqdm.tqdm(range(int(tolerance)), desc="P-agent"):
                p_steps += 1

                ques = P_agent.take_action(know, P_state, init_diff)
                init_diff = None

                next_P_state, observation, p_reward, done, init_ques, init_ans, q_diff = env.step_p(init_ques, init_ans, str(ques))
                logs_know_P.append("step{} select question {} and diff is {}".format(i, ques, q_diff))

                p_step_reward += p_reward
                P_agent.store_transition(Data_P(P_state.detach().numpy(), int(ques), p_reward, next_P_state.detach().numpy(), done))

                P_state = next_P_state
                logs_episode_P.append("learn after step {} question {} on know {} state {}".format(i,ques,know, P_state))

                if done:
                    print("Done!"+"{}".format(p_steps))
                    logs_episode_P.append("Done!"+"{}".format(p_steps))
                    if target and know in target:
                        target.remove(know)
                    break


            logs_episode_P.append("tolerance is {}, practice {} times".format(tolerance, p_steps))


            if P_agent.memory_counter >= batch_size:
                P_agent.learn()

            observation, l_reward, done, info = env.step_l()

            next_L_state = retrive_state(next_P_state, env.know_item)
            # L_rewards.append(l_reward)

            if l_steps >= L_max_steps:
                print("Learning Max steps!"+"{}".format(l_steps))
                logs_episode_L.append("Learning Max steps!"+"{}".format(l_steps))
                break

            if done or len(target)==0:
                print("Learning success!"+"{}".format(l_steps))
                break

            L_agent.store_transition(Data_L(L_state.detach().numpy(), int(know), l_reward, next_L_state.detach().numpy(), done))

            L_state = next_L_state
            last_tor = int(tolerance)
            last_prac_num = p_steps



        if L_agent.memory_counter >= batch_size:
            L_agent.learn()

        observation, l_reward, done, info = env.end_episode()
        L_rewards.append(l_reward)

        print('Episode: {} | Episode Reward: {:.2f}'.format(
            episode, l_reward))

        # logs_episode.append(steps)
        logs_episode_L.append("learned knowledges: ".format(l_knows))
        logs_episode_L.append(init_ans)

        all_logs_episode_L.append(logs_episode_L)


    print("all_logs_episode:",all_logs_episode_L)
    np.savetxt("all_logs_episode.txt", all_logs_episode_L, fmt="%s", delimiter="\n")
    print("all_logs_episode saved!")

    return L_rewards




if __name__ == '__main__':


    env = gym.make("KSS-v2")
    state_dim = env.action_space.n
    action_dim = env.action_space.n
    hidden_dim = 128
    actor_lr = 0.001
    critic_lr = 0.001
    gamma = 0.98

    lmbda = 0.95
    epochs = 10
    eps = 0.2

    max_episode_num = 30
    batch_size = 128


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    L_agent = PPO(state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 lmbda, epochs, eps, gamma, device,batch_size)

    P_agent = ActorCritic(state_dim, hidden_dim, action_dim, actor_lr,
                        critic_lr, gamma, env.learning_item_base.knowledge2item, device,batch_size)

    rewards = train(env, L_agent, P_agent, max_episode_num, batch_size)

    print(np.sum(rewards) / len(rewards))

    plt.plot(rewards)
    plt.show()


