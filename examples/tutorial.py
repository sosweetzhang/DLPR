import gym
from EduSim.Envs.KSS import KSSAgent

env = gym.make('KSS-v2')
agent = KSSAgent(env.action_space)
max_episode_num = 1000
n_step = False
max_steps = 20
train = True

episode = 0

while True:
    if max_episode_num is not None and episode > max_episode_num:
        break

    try:
        agent.begin_episode(env.begin_episode())
        episode += 1
    except ValueError:  # pragma: no cover
        break

    # recommend and learn
    if n_step is True:
        # generate a learning path
        learning_path = agent.n_step(max_steps)
        env.n_step(learning_path)
    else:
        # generate a learning path step by step
        for _ in range(max_steps):
            try:
                learning_item = agent.step()
            except ValueError:  # pragma: no cover
                break
            # interaction, _ ,_, _ = env.step(learning_item)
            # agent.observe(**interaction["performance"])
            interaction = env.step(learning_item)
            agent.observe(*interaction)

    # test the learner to see the learning effectiveness
    agent.episode_reward(env.end_episode()["reward"])
    agent.end_episode()

    if train is True:
        agent.tune()