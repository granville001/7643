"""
S2S Encoder model.  (c) 2021 Georgia Tech

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


class Encoder(nn.Module):
    """ The Encoder module of the Seq2Seq model
        You will need to complete the init function and the forward function.
    """

    def __init__(self, input_size, emb_size, encoder_hidden_size, decoder_hidden_size, dropout=0.2, model_type="RNN"):
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.emb_size = emb_size
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.model_type = model_type
        #############################################################################
        # TODO:                                                                     #
        #    Initialize the following layers of the encoder in this order!:         #
        #       1) An embedding layer                                               #
        #       2) A recurrent layer, this part is controlled by the "model_type"   #
        #          argument. You need to support the following type(in string):     #
        #          "RNN" and "LSTM".                                                #
        #       3) Linear layers with ReLU activation in between to get the         #
        #          hidden weights of the Encoder(namely, Linear - ReLU - Linear).   #
        #          The size of the output of the first linear layer is the same as  #
        #          its input size.                                                  #
        #          HINT: the size of the output of the second linear layer must     #
        #          satisfy certain constraint relevant to the decoder.              #
        #       4) A dropout layer                                                  #
        #                                                                           #
        # NOTE: Use nn.RNN and nn.LSTM instead of the naive implementation          #
        #############################################################################

        self.embedding = nn.Embedding(input_size, emb_size)

        if model_type == 'RNN':
            self.recurrentLayer = nn.RNN(emb_size, encoder_hidden_size, batch_first=True)
            self.linear1 = nn.Linear(encoder_hidden_size, encoder_hidden_size)
            self.linear2 = nn.Linear(encoder_hidden_size, decoder_hidden_size)
        elif model_type == 'LSTM':
            self.recurrentLayer = nn.LSTM(emb_size, encoder_hidden_size, batch_first=True)
            self.linear1 = nn.Linear(
                encoder_hidden_size, encoder_hidden_size)
            self.linear2 = nn.Linear(
                encoder_hidden_size, decoder_hidden_size)
        else:
            print("Recurrent unit not found")
            self.recurrentLayer = None

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        self.tanh = nn.Tanh()

        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

    def forward(self, input):
        """ The forward pass of the encoder
            Args:
                input (tensor): the encoded sequences of shape (batch_size, seq_len)

            Returns:
                output (tensor): the output of the Encoder; later fed into the Decoder.
                hidden (tensor): the weights coming out of the last hidden unit
        """

        #############################################################################
        # TODO: Implement the forward pass of the encoder.                          #
        #       Apply the dropout to the embedding layer before you apply the       #
        #       recurrent layer                                                     #
        #       Apply tanh activation to the hidden tensor before returning it      #
        #############################################################################


        x = self.embedding(input)
        x = self.dropout(x)

        if self.model_type == 'RNN':
            x, h = self.recurrentLayer(x)
            hidden = h
            hidden = self.linear1(hidden)
            hidden = self.relu(hidden)
            hidden = self.linear2(hidden)
            hidden = self.tanh(hidden)
        else:
            x, (h, c) = self.recurrentLayer(x)
            h = self.linear1(h)
            h = self.relu(h)
            h = self.linear2(h)
            h = self.tanh(h)
            hidden = torch.cat((h, c), dim=-1)
        output = x.permute((0, 1, 2))
        


        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return output, hidden

