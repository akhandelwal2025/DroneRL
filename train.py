from utils import *
import config
from Drone import Drone
from networks import *


def train():
    # initialize policy, value networks
    policy = Policy(input=config.STATE_DIM,
                    hidden=[128, 128],
                    output=config.ACTION_DIM)

    value = FeedForwardNetwork(input=config.STATE_DIM, 
                                hidden=[128, 128], 
                                output=1)
    
    # Create Drone instance
    env = Drone(init_x=Vector3(0, 0, 10), 
                init_theta=Vector3(0, 0, 0),
                init_v=Vector3(0, 0, 0),
                init_omega=Vector3(0, 0, 0),
                init_a=Vector3(0, 0, 0),
                init_alpha=Vector3(0, 0, 0)) # z=+10
    
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
                policy.forward(state)
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
            states = eps.get_states() # np.ndarray (Nx18, N = num states, 18 = num elements defining pose - x, v, a, theta, omega, alpha)
            values = value.forward(states) # (Nx1)
            eps.compute_advantages(values)

        # step 6: update the policy by maximizing the PPO-Clip objective
        
        



        

                


if __name__ == "__main__":
    train()