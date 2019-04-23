#collaborators: Sushanti Prabhu, Deepthi Devaraj 
from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        observations = [self.obs_dict[i] for i in Osequence]
        
        for s in range(S):
            alpha[s, 0] = self.pi[s] * self.B[s][observations[0]]
        for time in range(1,L):
            for st in range(S):
                alpha[st][time] =  sum([self.A[k][st] * alpha[k][time-1] for k in range(S)]) * self.B[st][observations[time]] 
        
      
        #return alpha
        ###################################################
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        observations = [self.obs_dict[i] for i in Osequence]
        for st in range(S):
            beta[st, L-1] = 1
        for time in range(L-2, -1, -1):
            for st in range(S):
                beta[st, time] = sum([self.A[st, i] * self.B[i, observations[time+1]] * beta[i,time+1] for i in range(S)])
        
        #return beta
        ###################################################
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        alphas=self.forward(Osequence)
        prob=sum(alphas[:,-1]);
        ###################################################
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ans=np.empty((0))
        alphas=self.forward(Osequence)
        betas=self.backward(Osequence)
        sequence_probability=self.sequence_prob(Osequence)
        
        ans=(alphas*betas)/sequence_probability
        ###################################################
        return ans

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        states = self.pi
        trans_prob = self.A
        obser_prob = self.B
        observations=[]
        for i in range(len(Osequence)):
            observations.append(self.obs_dict[Osequence[i]])
        
        int_st = {}
        for s in self.state_dict:
            int_st[self.state_dict[s]]=s
        
        fkh={}
        viterbi_path=[]
        sr_int = {}
        trans_ej=[]
        
        
        viterbi_path.append(fkh)

        #for j in range(len(trans_prob )):
        j=0
        while j<len(trans_prob):
              trans_ej.append(j)
              j+=1

        q=observations[0]
        ds=0
        
        while ds<len(trans_ej):
            sr_int[trans_ej[ds]] = [trans_ej[ds]]
            viterbi_path[0][trans_ej[ds]] = np.multiply(obser_prob[trans_ej[ds],q],states[trans_ej[ds]])
            ds+=1


        
        for jk in range(1, len(observations)):
            path_n = {}
            viterbi_path.append({})           
            for ds in trans_ej:
                u_j=[]
                for dso in trans_ej:
                    q=observations[jk]
                    u_j.append(viterbi_path[jk-1][dso] * trans_prob [dso][ds] * obser_prob[ds][q])
                vu=np.array(u_j)
                gds=np.max(vu)
                gfh=np.argmax(vu)
                viterbi_path[jk][ds] = gds
                path_n[ds] = [ds] + sr_int[gfh]  
            sr_int = path_n
            lam = 0    
            if len(observations)!=1:
                lam = jk
          

        (gds, gfh) = max((viterbi_path[lam][d], d) for d in trans_ej)
        
        path=sr_int[gfh]
        path=path[::-1]
        
        final_path=[]
        
        idx=0
        while idx<len(path):
            final_path.append(str(int_st[path[idx]]))
            idx+=1
            
        path = final_path
        ###################################################
        return path
