import torch
from utils import *

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, input, hidden, output):
        # create input layer
        self.model = [torch.nn.Linear(input, hidden[0]), 
                      torch.nn.Tanh()]
        
        # add hidden layers
        for i in range(1, len(hidden)):
            self.model.append(torch.nn.Linear(hidden[i-1], hidden[i]))
            self.model.append(torch.nn.Tanh())

        # add output layer
        self.model.append(torch.nn.Linear(hidden[-1], output))
        self.model = torch.nn.Sequential(self.model)
        
    def forward(self, input: Pose):
        input = input.to_numpy()
        return self.model(input)

class Policy(FeedForwardNetwork):
    def __init__(self, input, hidden, action_dim):
        super.__init__(input, hidden, action_dim)
        self.model.add_module("final_softmax", torch.nn.Softmax()) # adding softmax at output to constrain thrust outputs between 0-1

        self.action = np.zeros(action_dim)
        self.log_std = torch.nn.Parameter(np.zeros(action_dim))
    
    def generate_action_log_prob(self):
        # need to generate action for each rotor + return log prob for that action
        # actions are sampled from gaussian distribution w/ mean, std
        # mean for each rotor is defined by policy mlp output
        # std is learnable parameter separate from mlp that is updated through backward pass b/c it is involved in loss function #TODO is that correct?
        # log prob is calculated from PDF function of gaussian distribution

        # Derivation of log prob formula
        # PDF of Gaussian Distribution (aka prob of selecting an element - f(x | mu, sigma)) = 1/(sigma * sqrt(2*pi)) * exp(-((x - mu) ** 2)/(2*sigma**2))
        # log-prob = log of PDF = log(f(x | mu, sigma)) = -log(sigma) - 1/2*log(2 * pi) - ((x - mu) ** 2)/(2 * sigma ** 2)

        sampled_actions = []
        log_probs = []
        for i in range(4):
            mean = self.action[i] # mu
            std = torch.exp(self.log_std)[0] # sigma
            sampled_action = np.random.normal(loc=self.action[i], scale=torch.exp(self.log_std)) #loc = mean/mu, scale = std/sigma
            sampled_action = np.clip(sampled_action, 0, 1) # clip sampled action to [0, 1]
            log_prob = -np.log(std) - 0.5*np.log(2 * np.pi) - ((sampled_action - mean) ** 2/(2 * (std ** 2)))
            sampled_actions.append(sampled_action)
            log_probs.append(log_prob)
        
        self.action = np.asarray(sampled_actions)
        self.log_probs = np.asarray(log_probs)
        return self.action, self.log_probs