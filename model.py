import torch
import torch.nn as nn
import torch.nn.functional as F
from edl_pytorch import NormalInvGamma

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
'''
class BM_sep_EDR_NN(nn.Module):
    def __init__(self, in_size = 2,hidden_size = 1024,out_size = 2,n_comp=20,embed = 50,drop_p=0.1):
        super(BM_sep_EDR_NN, self).__init__()
        self.hidden = hidden_size
        hidden =self.hidden
        self.n_comp = n_comp
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size
        self.drop_p = drop_p
        self.encoder = nn.Sequential(
            nn.Linear(in_size, hidden),
            # nn.ReLU(),
            nn.Softplus(),
            # nn.Tanh(),
            nn.Linear(hidden,embed),
        )
        self.decoder1 = nn.Sequential(
            nn.Linear(embed, hidden),
            # nn.ReLU(),
            nn.Softplus(),
            # nn.Tanh(),

            # nn.Linear(hidden, hidden),
            # nn.Softplus(),
            # # nn.Tanh(),
            # # nn.ReLU(),
            # nn.Linear(hidden, hidden),
            # nn.Softplus(),
            # # nn.ReLU(),
            # # nn.Tanh(),

            # nn.Linear(hidden, hidden),
            # nn.Softplus(),
            nn.Dropout(p=self.drop_p),
            
            # nn.Tanh(),
            # nn.ReLU(),
            # nn.Linear(hidden,(2*out_size+1)*n_comp),
            # nn.ReLU(),
            # nn.Linear(embed,(2*out_size+1)*n_comp),
            NormalInvGamma(hidden,n_comp),

            # nn.ReLU(),
            # nn.Softplus(),
            # nn.Dropout(p=0.2),
        )
        self.decoder2 = nn.Sequential(
            nn.Linear(embed, hidden),
            # nn.ReLU(),
            nn.Softplus(),
            # nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Softplus(),
            # nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.Softplus(),
            # nn.ReLU(),
            # nn.Tanh(),

            nn.Linear(hidden, hidden),
            nn.Softplus(),
            nn.Dropout(p=self.drop_p),
            
            # nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(hidden,(2*out_size)*n_comp),
            nn.Softplus(),
            # nn.ReLU(),
            # nn.Linear(embed,(2*out_size+1)*n_comp),
            # NormalInvGamma(hidden,2*out_size*n_comp),

            # nn.ReLU(),
            # nn.Softplus(),
            # nn.Dropout(p=0.2),
        )
        # self.comp_0 = torch.ones((n_comp,out_size))
        self.comp_0 = torch.ones(2*n_comp*out_size).to(device)
        self.comp_1 = torch.zeros(2*n_comp*out_size).to(device)
    def update_comp(self,comp):
        self.comp_0 = self.comp_0+torch.tensor(comp,dtype=torch.float32).to(device)
    def deupdate_comp(self,comp):
        self.comp_0 = self.comp_0-torch.tensor(comp,dtype=torch.float32).to(device)
    def compute_cov(self):
        print(self.comp_0.shape)
        a = self.comp_0[:,:self.n_comp*self.out_size].view(self.n_comp,self.out_size)
        b = self.comp_0[:,self.n_comp*self.out_size:].view(self.n_comp,self.out_size)
        theta = a/(a+b)
        self.lamb_k = []
        self.theta_G = []
        for k in range(self.n_comp):
            self.lamb_k.append(torch.diag(theta[k]*(1-theta[k])))
            # print('lambk',self.lamb_k[-1].shape)
            self.theta_G.append(self.lamb_k[-1]+torch.matmul(theta[k],theta[k].T))
            # print('theta_G',self.theta_G[-1].shape)
        return self.theta_G

    
    def forward(self, data):
        emb = self.encoder(data)
        pred_pi = self.decoder1(emb)
        pred_comp = self.decoder2(emb)
        # print(pred.dim)
        # if pred.dim()==1:
        #     pred = pred.unsqueeze(0)
        # print(pred.dim())
        pi_mu, pi_v, pi_alpha, pi_beta = (d.squeeze() for d in pred_pi)
        # pi = F.ReLU(pi_mu)
        pi = pi_mu

        # theta_mu, comp_v, comp_alpha, comp_beta = (d.squeeze() for d in pred_comp)
        # theta = F.ReLU(theta_mu)
        theta=pred_comp
        output = [pi,theta,pi_v, pi_alpha, pi_beta]
        
        return output
'''

class BM_sep_EDR_NN_custom(nn.Module):
    def __init__(self, in_size = 2,hidden_size = 1024,\
                out_size = 2,n_comp=20,embed = 50,\
                drop_p=0.1,activation = 'softplus',\
                decoder_type = 'deep', cons = False):
        super(BM_sep_EDR_NN_custom, self).__init__()
        self.hidden = hidden_size
        hidden =self.hidden
        self.n_comp = n_comp # number of components
        self.embed = embed
        self.in_size = in_size
        self.out_size = out_size # number of labels
        self.drop_p = drop_p
        self.activation = activation # use softplus unless specified
        self.decoder_type = decoder_type # test deep and shallow 
        self.cons = cons # whether softplus pi
        if(self.activation =='softplus'):
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.Softplus(),
                nn.Linear(hidden,embed),
            )
            self.decoder1 = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.Softplus(),
                nn.Dropout(p=self.drop_p),
                NormalInvGamma(hidden,n_comp),
            )
            if (decoder_type == 'deep') :
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Linear(hidden, hidden),
                    nn.Softplus(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.Softplus(),
                )
            else:
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.Softplus(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.Softplus(),
                )
        elif(self.activation =='tanh'):
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.Tanh(),
                nn.Linear(hidden,embed),
            )
            self.decoder1 = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.Tanh(),
                nn.Dropout(p=self.drop_p),
                NormalInvGamma(hidden,n_comp),
            )
            if (decoder_type == 'deep') :
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Linear(hidden, hidden),
                    nn.Tanh(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.Tanh(),
                )
            else:
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.Tanh(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.Tanh(),
                )
        elif(self.activation =='relu'):
            self.encoder = nn.Sequential(
                nn.Linear(in_size, hidden),
                nn.ReLU(),
                nn.Linear(hidden,embed),
            )
            self.decoder1 = nn.Sequential(
                nn.Linear(embed, hidden),
                nn.ReLU(),
                nn.Dropout(p=self.drop_p),
                NormalInvGamma(hidden,n_comp),
            )
            if (decoder_type == 'deep') :
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Linear(hidden, hidden),
                    nn.ReLU(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.ReLU(),
                )
            else:
                self.decoder2 = nn.Sequential(
                    nn.Linear(embed, hidden),
                    nn.ReLU(),
                    nn.Dropout(p=self.drop_p),
                    nn.Linear(hidden,(2*out_size)*n_comp),
                    nn.ReLU(),
                )

        self.comp_0 = torch.ones(2*n_comp*out_size).to(device) # components, udpated outside when initialized
        self.comp_1 = torch.zeros(2*n_comp*out_size).to(device) # initilized as zeros for later use
    def update_comp(self,comp):
        self.comp_0 = self.comp_0+torch.tensor(comp,dtype=torch.float32).to(device)
    def deupdate_comp(self,comp):
        self.comp_0 = self.comp_0-torch.tensor(comp,dtype=torch.float32).to(device)
    def compute_cov(self):
        # components convariance for sampling
        print(self.comp_0.shape)
        # get positive and negative pseudo counts (beta parameters)
        a = self.comp_0[:,:self.n_comp*self.out_size].view(self.n_comp,self.out_size)
        b = self.comp_0[:,self.n_comp*self.out_size:].view(self.n_comp,self.out_size)
        # Bernoulli parameter
        theta = a/(a+b)
        self.lamb_k = []
        self.theta_G = []
        for k in range(self.n_comp):
            self.lamb_k.append(torch.diag(theta[k]*(1-theta[k])))
            # print('lambk',self.lamb_k[-1].shape)
            self.theta_G.append(self.lamb_k[-1]+torch.matmul(theta[k],theta[k].T))
            # print('theta_G',self.theta_G[-1].shape)
        return self.theta_G

    
    def forward(self, data):
        # common embeddings
        emb = self.encoder(data)
        # prediction of weight assignments pi from decoder1
        pred_pi = self.decoder1(emb)
        # prediction of instance-wise component adjustment from decoder2
        pred_comp = self.decoder2(emb)

        pi_mu, pi_v, pi_alpha, pi_beta = (d.squeeze() for d in pred_pi)
        if (self.cons == False):
            pi = pi_mu
        else:
            pi = F.ReLU(pi_mu)
        theta=pred_comp
        output = [pi,theta,pi_v, pi_alpha, pi_beta]        
        return output


