# Desc: Actor-Critic
# conding:utf-8

import torch
import torch.nn.functional as F
import sys
import random
import collections

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Data_P:
    def __init__(self, P_state, ques, p_reward, next_P_state, done):
        self.P_state = P_state
        self.ques = ques
        self.p_reward = p_reward
        self.next_P_state = next_P_state
        self.done = done


class Memory:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def set(self, data):
        self.buffer.append(data)


    def get(self, batch_size,device):
        mini_batch = random.sample(self.buffer, batch_size)
        state = torch.tensor([data.P_state for data in mini_batch], dtype=torch.float).to(device)
        action = torch.tensor([data.ques for data in mini_batch], dtype=torch.long).to(device)
        reward = torch.tensor([data.p_reward for data in mini_batch], dtype=torch.float).to(device)
        next_state = torch.tensor([data.next_P_state for data in mini_batch], dtype=torch.float).to(device)
        done = torch.tensor([data.done for data in mini_batch], dtype=torch.float).to(device)
        return state, action, reward, next_state, done


class ActorCritic:
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr,
                 gamma, know_item, device,batch_size):

        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.critic = ValueNet(state_dim, hidden_dim).to(device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)
        self.gamma = gamma
        self.device = device

        self.know_item = know_item

        self.memory_counter = 0
        self.memory = Memory(capacity=5000)

        self.batch_size = batch_size


    def candidate_ques(self,know, init_diff):
        try:
            all_ques = self.know_item[know]
        except:
            all_ques = self.know_item[know[0]]

        closest_difficulty = all_ques[0]
        for item in all_ques:
            if abs(item.difficulty - init_diff) < abs(closest_difficulty.difficulty - init_diff):
                closest_difficulty = item
        return closest_difficulty

    def take_action(self, know, state, init_diff=None):
        if init_diff is not None:
            ques = self.candidate_ques(know,init_diff)
            ques = int(ques.id)
        else:
            probs = self.actor(state)
            candidates = []
            try:
                all_ques = self.know_item[know]
            except:
                all_ques = self.know_item[know[0]]
            for item in all_ques:
                candidates.append(int(item.id))
            candidate_probs = probs.gather(dim=1, index=torch.Tensor(candidates).long().view(1, -1))
            ques = int(random.choices(candidates, weights=candidate_probs.view(-1).detach().numpy())[0])
        return ques


    def store_transition(self, data):
        self.memory.set(data)
        self.memory_counter += 1



    def learn(self):

        states, actions, rewards, next_states, dones = self.memory.get(self.batch_size,self.device)

        td_target = rewards + self.gamma * self.critic(next_states) * (1 -dones)
        td_delta = td_target - self.critic(states)
        log_probs = torch.log(torch.gather(self.actor(states).squeeze(1), 1, actions.unsqueeze(-1).long()))

        actor_loss = torch.mean(-log_probs * td_delta.detach())

        critic_loss = torch.mean(
            F.mse_loss(self.critic(states), td_target.detach()))
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()