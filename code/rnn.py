# coding: utf-8
from rnnmath import *
from model import Model, is_param, is_delta

class RNN(Model):
    '''
    This class implements Recurrent Neural Networks.
    
    You should implement code in the following functions:
        predict				->	predict an output sequence for a given input sequence
        acc_deltas			->	accumulate update weights for the RNNs weight matrices, standard Back Propagation
        acc_deltas_bptt		->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time
        acc_deltas_np		->	accumulate update weights for the RNNs weight matrices, standard Back Propagation -- for number predictions
        acc_deltas_bptt_np	->	accumulate update weights for the RNNs weight matrices, using Back Propagation Through Time -- for number predictions

    Do NOT modify any other methods!
    Do NOT change any method signatures!
    '''
    
    def __init__(self, vocab_size, hidden_dims, out_vocab_size):
        '''
        initialize the RNN with random weight matrices.
        
        DO NOT CHANGE THIS
        
        vocab_size		size of vocabulary that is being used
        hidden_dims		number of hidden units
        out_vocab_size	size of the output vocabulary
        '''

        super().__init__(vocab_size, hidden_dims, out_vocab_size)

        # matrices V (input -> hidden), W (hidden -> output), U (hidden -> hidden)
        with is_param():
            self.U = np.random.randn(self.hidden_dims, self.hidden_dims)*np.sqrt(0.1)
            self.V = np.random.randn(self.hidden_dims, self.vocab_size)*np.sqrt(0.1)
            self.W = np.random.randn(self.out_vocab_size, self.hidden_dims)*np.sqrt(0.1)

        # matrices to accumulate weight updates
        with is_delta():
            self.deltaU = np.zeros_like(self.U)
            self.deltaV = np.zeros_like(self.V)
            self.deltaW = np.zeros_like(self.W)

    def predict(self, x):
        '''
        predict an output sequence y for a given input sequence x
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        
        returns	y,s
        y	matrix of probability vectors for each input word
        s	matrix of hidden layers for each input word
        
        '''
        
        # matrix s for hidden states, y for output states, given input x.
        # rows correspond to times t, i.e., input words
        # s has one more row, since we need to look back even at time 0 (s(t=0-1) will just be [0. 0. ....] )

        s = np.zeros((len(x) + 1, self.hidden_dims))
        y = np.zeros((len(x), self.out_vocab_size))

        for t in range(len(x)):
            ##########################
            # --- your code here --- #
            ##########################
            x_t = make_onehot(x[t], self.vocab_size)
            net_in_t = self.V @ x_t + self.U @ s[t-1] # s[0] must always be [0, 0, ... , 0]
            s[t] = sigmoid(net_in_t)
            net_out_t = self.W @ s[t]
            y[t] = softmax(net_out_t)

        return y, s
    
    def acc_deltas(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x	list of words, as indices, e.g.: [0, 4, 2]
        d	list of words, as indices, e.g.: [4, 2, 3]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''

        for t in reversed(range(len(x))):
            ##########################
            # --- your code here --- #
            ##########################
            ##########################
            d_t = make_onehot(d[t], self.out_vocab_size)
            x_t = make_onehot(x[t], self.vocab_size)
            derivative_f_t = s[t] * (np.ones(len(s[t])) - s[t])
            # derivative_g_t = np.ones(self.out_vocab_size)
            delta_out_t =  (d_t - y[t])    #== (d_t - y[t]) * derivative_g_t
            delta_in_t = (self.W.T @ delta_out_t) * derivative_f_t

            #update W, V, U
            self.deltaW += np.outer(delta_out_t, s[t])
            self.deltaV += np.outer(delta_in_t, x_t)
            self.deltaU += np.outer(delta_in_t, s[t-1])



    def acc_deltas_np(self, x, d, y, s):
        '''
        accumulate updates for V, W, U
        standard back propagation
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y	predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
            should be part of the return value of predict(x)
        s	predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
            should be part of the return value of predict(x)
        
        no return values
        '''
        pass

        ##########################
        # --- your code here --- #
        ##########################
        
    def acc_deltas_bptt(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        
        x		list of words, as indices, e.g.: [0, 4, 2]
        d		list of words, as indices, e.g.: [4, 2, 3]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''
        for t in reversed(range(len(x))):
            d_t = make_onehot(d[t], self.out_vocab_size)
            derivative_f_t = s[t] * (np.ones(len(s[t])) - s[t])
            delta_out_t = (d_t - y[t])
            # No need to compute derivative_g_t since it is always 1:
            # derivative_g_t = np.ones(self.out_vocab_size)
            # => delta_out_t == delta_out_t * derivative_g_t

            # Update W once for timestep t
            self.deltaW += np.outer(delta_out_t, s[t])

            # Compute delta_in for timestep t
            delta_in = (self.W.T @ delta_out_t) * derivative_f_t

            ##self.deltaV += np.outer(delta_in, x_t_tau) #(if loop starts from 1)
            ##self.deltaU += np.outer(delta_in, s[t_tau - 1]) #(if loop starts from 1)

            # Backpropagate through time from timestep t-1 to t-'steps'
            for tau in range(0, steps + 1):
                t_tau = t - tau
                if t_tau < 0:
                    break
                x_t_tau = make_onehot(x[t_tau], self.vocab_size)
                self.deltaV += np.outer(delta_in, x_t_tau)  ##this should be at the end of the loop (if loop starts from 1)
                self.deltaU += np.outer(delta_in, s[t_tau - 1]) ##this should be at the end of the loop (if loop starts from 1)
                if tau == steps:
                    # no need to calculate delta_in for t-steps-1
                    break
                # Compute new delta_in
                derivative_f_t_tau = s[t_tau-1] * (np.ones(len(s[t_tau-1])) - s[t_tau-1])
                delta_in = (self.U.T @ delta_in) * derivative_f_t_tau
            ##########################
            # --- your code here --- #
            ##########################


    def acc_deltas_bptt_np(self, x, d, y, s, steps):
        '''
        accumulate updates for V, W, U
        back propagation through time (BPTT)
        
        this should not update V, W, U directly. instead, use deltaV, deltaW, deltaU to accumulate updates over time
        for number prediction task, we do binary prediction, 0 or 1

        x	list of words, as indices, e.g.: [0, 4, 2]
        d	array with one element, as indices, e.g.: [0] or [1]
        y		predicted output layer for x; list of probability vectors, e.g., [[0.3, 0.1, 0.1, 0.5], [0.2, 0.7, 0.05, 0.05] [...]]
                should be part of the return value of predict(x)
        s		predicted hidden layer for x; list of vectors, e.g., [[1.2, -2.3, 5.3, 1.0], [-2.1, -1.1, 0.2, 4.2], [...]]
                should be part of the return value of predict(x)
        steps	number of time steps to go back in BPTT
        
        no return values
        '''
        pass

        ##########################
        # --- your code here --- #
        ##########################