"""
S2S Decoder model.  (c) 2021 Georgia Tech

Copyright 2021, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 7643 Deep Learning

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as Github, Bitbucket, and Gitlab.  This copyright statement should
not be removed or edited.

Sharing solutions with current or future students of CS 7643 Deep Learning is
prohibited and subject to being investigated as a GT honor code violation.

-----do not edit anything above this line---
"""

import random

import torch
import torch.nn as nn
import torch.optim as optim


class Decoder(nn.Module):
    """ The Decoder module of the Seq2Seq model 
        You will need to complete the init function and the forward function.
    """

    def __init__(self, emb_size, encoder_hidden_size, decoder_hidden_size, output_size, dropout=0.2, model_type="RNN"):
        super(Decoder, self).__init__()

        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.output_size = output_size
        self.model_type = model_type

        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the decoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN", "LSTM".                                                   #
        #       3) A single linear layer with a (log)softmax layer for output       #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################
        self.embedding = nn.Embedding(output_size, emb_size)

        if model_type == 'RNN':
            self.recurrentLayer = nn.RNN(emb_size, decoder_hidden_size, batch_first=True)
            self.linear1 = nn.Linear(decoder_hidden_size, output_size)
        elif model_type == 'LSTM':
            self.recurrentLayer = nn.LSTM(emb_size, decoder_hidden_size, batch_first=True)
            self.linear1 = nn.Linear(decoder_hidden_size, output_size)
        else:
            print("Model type not found")

        
        self.dropout = nn.Dropout(p=dropout)
        self.logsoftmax = nn.LogSoftmax(dim=2)

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def compute_attention(self, hidden, encoder_outputs):
        """ compute attention probabilities given a controller state (hidden) and encoder_outputs using cosine similarity
            as your attention function.

                cosine similarity (q,K) =  q@K.Transpose / |q||K|
                hint |K| has dimensions: N, T
                Where N is batch size, T is sequence length

            Args:
                hidden (tensor): the controller state (dimensions: 1,N, hidden_dim)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention (dimensions: N,T, hidden dim)
            Returns:
                attention: attention probabilities (dimension: N,1,T)
        """
        import torch.nn.functional as F
        #############################################################################
        #                              BEGIN YOUR CODE                              #
        # It is recommended that you implement the cosine similarity function from  #
        # the formula given in the docstring. This exercise will build up your     #
        # skills in implementing mathematical formulas working with tensors.        #
        # Alternatively you may use nn.torch.functional.cosine_similarity or        #
        # some other similar function for your implementation.                      #
        #############################################################################
        # Compute dot product between hidden state and encoder outputs transposed
        dot_product = torch.bmm(hidden.transpose(0,1), encoder_outputs.transpose(1, 2))
    
        # Compute norms of hidden state and encoder outputs
        hidden_norm = torch.norm(hidden, dim=-1, keepdim=True)
        encoder_outputs_norm = torch.norm(encoder_outputs, dim=-1, keepdim=True)
    
        # Ensure encoder_outputs_norm has shape (N, T, 1) instead of (N, T)
        encoder_outputs_norm = encoder_outputs_norm.transpose(1, 2)
    
        # Compute cosine similarity
        cosine_similarity = dot_product / (hidden_norm * encoder_outputs_norm)
    
        # Softmax along the last dimension to obtain attention probabilities
        attention = F.softmax(cosine_similarity, dim=-1)
        
        attention_list = []
        
        # Iterate over the indices and extract the elements
        for i in range(attention.shape[0]):
            attention_list.append(attention[i, i, 0:attention.shape[2]].unsqueeze(0).unsqueeze(0))  # Extract and append the elements
        
        # Concatenate the extracted elements into a single tensor
        attention = torch.cat(attention_list, dim=0)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return attention


    def forward(self, input, hidden, encoder_outputs=None, attention=False):
        """ The forward pass of the decoder
            Args:
                input (tensor): the encoded sequences of shape (N, 1). HINT: encoded does not mean from encoder!!
                hidden (tensor): the hidden state of the previous time step from the decoder, dimensions: (1,N,decoder_hidden_size)
                encoder_outputs (tensor): the outputs from the encoder used to implement attention, dimensions: (N,T,encoder_hidden_size)
                attention (Boolean): If True, need to implement attention functionality
            Returns:
                output (tensor): the output of the decoder, dimensions: (N, output_size)
                hidden (tensor): the state coming out of the hidden unit, dimensions: (1,N,decoder_hidden_size)
                where N is the batch size, T is the sequence length
        """
        
        #############################################################################
        # TODO: Implement the forward pass of the decoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #                                                                           #
        #       If attention is true, compute the attention probabilities and use   #
        #       them to do a weighted sum on the encoder_outputs to determine       #
        #       the hidden (and cell if LSTM) states that will be consumed by the   #
        #       recurrent layer.                                                    #
        #                                                                           #
        #       Apply linear layer and log-softmax activation to output tensor      #
        #       before returning it.                                                #
        #############################################################################
        x = self.embedding(input)
        x = self.dropout(x)
        

        
        if self.model_type == 'RNN':
            
            if attention == True:
                
                att = self.compute_attention(hidden, encoder_outputs)
                weighted_sum = torch.sum(torch.bmm(att, encoder_outputs),dim=1).reshape(hidden.shape) # Assuming encoder_outputs is a tensor
                hidden = weighted_sum
                
            output, hidden = self.recurrentLayer(x, hidden)
            
        elif self.model_type == 'LSTM':
            
            h = hidden[:, :, :self.decoder_hidden_size].contiguous()
            c = hidden[:, :, self.decoder_hidden_size:].contiguous()
            
            if attention == True:

                
                atth = self.compute_attention(h, encoder_outputs)
                attc = self.compute_attention(c, encoder_outputs)
                
                weighted_sum_h = torch.sum(torch.bmm(atth, encoder_outputs),dim=1).reshape(h.shape) 
                weighted_sum_c = torch.sum(torch.bmm(attc, encoder_outputs),dim=1).reshape(c.shape) # Assuming encoder_outputs is a tensor
                
                h = weighted_sum_h
                c = weighted_sum_c
            
            output, (h, c) = self.recurrentLayer(x, (h, c))
            hidden = torch.cat((h, c), dim=-1)

        else:
             print("Model type not found")   

        output = self.linear1(output)
        output = self.logsoftmax(output)
        output = output[:, 0, :]


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return output, hidden
       
