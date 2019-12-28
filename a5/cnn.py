#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch


class CNN(torch.nn.Module):
    def __init__(self, e_char, e_word,  m_word, k=5):
        """
        Init the CNN layer.
        @param e_char (int): Dimensionality of character embeddings. Also number of input channels.
        @param e_word (int): Dimensionality of word embeddings. Also number of filters / output channels.
        @param m_word (int): Predefined hyperparameter representing maximum word length.
        @param k (int): Kernel size.
        """
        super(CNN, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels=e_char, out_channels=e_word, kernel_size=k)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool1d(kernel_size=m_word-k+1)

    def forward(self, x_reshaped):
        """ Forward pass of a convolution layer. Takes a mini-batch as an argument.

        @param x_reshaped (Tensor): Input to the convolution contains character embeddings R,
                                    it's shape is (batch size, e_char, m_word).
        @returns x_conv_out (Tensor): a variable/tensor of shape (batch size, e_char)
        """
        x_conv_out = self.maxpool(self.relu(self.conv(x_reshaped)))
        x_conv_out = torch.squeeze(x_conv_out, dim=2)
        return x_conv_out

### END YOUR CODE

