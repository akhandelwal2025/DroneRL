import torch
from utils import *

class FeedForwardNetwork(torch.nn.Module):
    def __init__(self, input, hidden, output, init_lr):
        # create input layer
        super().__init__()
        self.model = [torch.nn.Linear(input, hidden[0]), 
                      torch.nn.Tanh()]
        
        # add hidden layers
        for i in range(1, len(hidden)):
            self.model.append(torch.nn.Linear(hidden[i-1], hidden[i]))
            self.model.append(torch.nn.Tanh())

        # add output layer
        self.model.append(torch.nn.Linear(hidden[-1], output))
        self.model = torch.nn.Sequential(*self.model)

        # store the output of the network
        self.output = np.zeros(output)

        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(),
                                           lr=init_lr)
        
    def forward_single_state(self, input: Pose):
        input = input.to_tensor()
        print(input.shape)
        self.output = self.model(input).detach().numpy()
        return self.output
    
    def forward_batch_states(self, input: torch.Tensor):
        # input.shape = (Nx18) - N = number of states, 18 = elements in pose
        # doing this batch update works because output will be of size (Nx4) - i.e. for each state, you get four rotor means
        # log_std stays the same for all states, it is only updated on backward pass, so sampling actions should be fine 
        self.output = self.model(input).detach().numpy()
        return self.output
    
    def backward(self, loss):
        if isinstance(loss, float):
            loss = torch.tensor(loss, dtype=torch.float32, requires_grad=True)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Policy(FeedForwardNetwork): 
    def __init__(self, input, hidden, action_dim, init_lr):
        super().__init__(input, hidden, action_dim, init_lr)
        self.model.add_module("final_softmax", torch.nn.Softmax()) # adding softmax at output to constrain thrust outputs between 0-1
        self.log_std = torch.nn.Parameter(torch.zeros(1))

        self.sampled_actions = np.empty(action_dim)
        self.sampled_log_probs = np.empty(action_dim)

    def generate_action_log_prob(self):
        # need to generate action for each rotor + return log prob for that action
        # actions are sampled from gaussian distribution w/ mean, std
        # mean for each rotor is defined by policy mlp output
        # std is learnable parameter separate from mlp that is updated through backward pass b/c it is involved in loss function #TODO is that correct?
        # log prob is calculated from PDF function of gaussian distribution

        # Derivation of log prob formula
        # PDF of Gaussian Distribution (aka prob of selecting an element - f(x | mu, sigma)) = 1/(sigma * sqrt(2*pi)) * exp(-((x - mu) ** 2)/(2*sigma**2))
        # log-prob = log of PDF = log(f(x | mu, sigma)) = -log(sigma) - 1/2*log(2 * pi) - ((x - mu) ** 2)/(2 * sigma ** 2)

        # self.output = Nx4, sampled_action = Nx4
        std = torch.exp(self.log_std).detach().numpy() # sigma
        self.sampled_actions = np.random.normal(loc=self.output, scale=std) #loc = mean/mu, scale = std/sigma
        self.sampled_actions = np.clip(self.sampled_actions, 0, 1) # clip sampled action to [0, 1]
        self.sampled_log_probs = -np.log(std) - 0.5*np.log(2 * np.pi) - ((self.sampled_actions - self.output) ** 2/(2 * (std ** 2)))
        return self.sampled_actions, self.sampled_log_probs
    
        # ----------- PREVIOUS IMPLEMENTATION -----------
        # sampled_actions = []
        # log_probs = []
        # for i in range(4):
        #     mean = self.action[i] # mu
        #     std = torch.exp(self.log_std)[0] # sigma
        #     sampled_action = np.random.normal(loc=self.action[i], scale=torch.exp(self.log_std)) #loc = mean/mu, scale = std/sigma
        #     sampled_action = np.clip(sampled_action, 0, 1) # clip sampled action to [0, 1]
        #     log_prob = -np.log(std) - 0.5*np.log(2 * np.pi) - ((sampled_action - mean) ** 2/(2 * (std ** 2)))
        #     sampled_actions.append(sampled_action)
        #     log_probs.append(log_prob)
        
        # self.action = np.asarray(sampled_actions)
        # self.log_probs = np.asarray(log_probs)
        # return self.action, self.log_probs