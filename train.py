from utils import *
import config
from Drone import Drone
from networks import *


def train():
    # initialize policy, value networks
    policy = Policy(input=config.STATE_DIM,
                    hidden=[128, 128],
                    action_dim=config.ACTION_DIM,
                    init_lr=config.INIT_LR)

    value = FeedForwardNetwork(input=config.STATE_DIM, 
                                hidden=[128, 128], 
                                output=1,
                                init_lr=config.INIT_LR)
    
    # Create Drone instance
    env = Drone(init_pose=config.init_pose,
                  target_pose=config.target_pose)
    
    def collect_batch(batch_num):
        batch = Batch(batch_num=batch_num,
                      eps_per_batch=config.EPISODES_PER_BATCH)

        for eps_num in range(config.EPISODES_PER_BATCH):
            eps = Episode(eps_num=eps_num,
                          discount_factor=config.DISCOUNT_FACTOR)

            state, done = env.reset()

            while not done:
                # add_state at the start adds very initial state and then adds all states as they are reached
                # terminal state is not added because the loop is broken. therefore terminal state not added
                eps.add_state(state)
                policy.forward_single_state(state)
                action, log_probs = policy.generate_action_log_prob()

                env.set_thrusts(*action)
                state, reward, done = env.step()
                
                eps.add_action(action)
                eps.add_reward(reward)
                eps.add_log_prob(log_probs)
    
            batch.add_eps(eps)

        return batch
    
    for batch_num in range(config.NUM_BATCHES):
        # step 3: collect set of trajectories D_k = {tau_i} by running policy pi_k = pi(theta_k) in the environment
        batch = collect_batch(batch_num)

        # step 4: compute rewards-to-go R-hat_t
        for eps in batch.episodes:
            eps.compute_discounted_reward_to_go()
        
        # step 5: compute advantage estimates A-hat_t based on the current value function V_phi_k
        # A_pi(s, a) = Q_pi(s, a) - V_phi_k(s)
        # Q_pi(s, a) = reward-to-gos (calculated in previous step)
        # V_phi_k(s) = value function in current iteration (k, parameters=phi), evaluated on every state
        for eps in batch.episodes:
            eps_states = eps.get_states() # np.ndarray (Nx18, N = num states, 18 = num elements defining pose - x, v, a, theta, omega, alpha)
            values = value.forward_batch_states(eps_states) # (Nx1)
            eps.compute_advantages(values)
        
        # step 6 + 7: update the policy by maximizing the PPO-Clip objective + fit value function by regression on MSE
        # PPO is inherently an on-policy algorithm, so theoretically, you should only update the policy/value mlp once with your current batch of samples
        # however since PPO enforces a trust region, thereby preventing too steep of a change in the policy per update, empirically it is found to work with slight off-policy
        # therefore, by running multiple updates on the same batch (multiple epochs), we are increasing sample efficiency by allowing the model to train off-policy
        batch.process_all_eps()
        for epoch in range(config.PPO_EPOCHS_PER_BATCH):
            policy.forward_batch_states(batch.all_states)
            _, new_log_probs = policy.generate_action_log_prob()
            
            old_log_probs = batch.all_log_probs
            ratios = np.exp(new_log_probs - old_log_probs)
            surr1 = ratios * batch.all_advantages
            surr2 = np.clip(ratios, 1+config.EPSILON, 1-config.EPSILON) * batch.all_advantages
            policy_loss = -np.min(surr1, surr2).mean()

            new_values = value.forward_batch_states(batch.all_states)  
            value_loss = torch.nn.functional.mse_loss(new_values, batch.all_rewards_to_go)

            policy.backward(policy_loss)
            value.backward(value_loss)

if __name__ == "__main__":
    train()