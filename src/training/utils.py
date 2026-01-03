import torch

def ppo_update(agent, state, action, log_prob, reward, next_state,
               gamma=0.99, clip_epsilon=0.2):

    state_t = torch.FloatTensor(state).unsqueeze(0)
    next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
    reward_t = torch.tensor(reward, dtype=torch.float32)

    value = agent.critic(state_t)
    next_value = agent.critic(next_state_t)
    advantage = reward_t + gamma * next_value - value

    new_log_prob, _, entropy = agent.evaluate(
        state_t, torch.tensor([action])
    )

    ratio = (new_log_prob - log_prob).exp()
    surr1 = ratio * advantage.detach()
    surr2 = torch.clamp(ratio, 1-clip_epsilon, 1+clip_epsilon) * advantage.detach()

    loss = -torch.min(surr1, surr2) + 0.5 * advantage.pow(2) - 0.01 * entropy

    agent.optimizer.zero_grad()
    loss.backward()
    agent.optimizer.step()
