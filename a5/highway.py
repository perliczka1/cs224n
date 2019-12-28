#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

import torch


class Highway(torch.nn.Module):
    def __init__(self, e_word):
        """
        Init the Highway layer. The paper with description: https://arxiv.org/pdf/1505.00387.pdf
        @param e_word (int): Embedding size - dimensionality of th input vector
        """
        super(Highway, self).__init__()
        self.W_proj = torch.nn.Linear(e_word, e_word, bias=True)
        self.W_out = torch.nn.Linear(e_word, e_word, bias=True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x_conv_out):
        """ Forward pass of a Highway layer. Takes a mini-batch as an argument.

        @param x_conv_out (Tensor): Output from convolution model, after max pooling was applied, it's shape is (batch size, e_word).
        @returns x_highway (Tensor): a variable/tensor of shape (batch size, e_word) representing the output from Highway layer.
                                    Dropout is not applied.
        """
        x_proj = self.relu(self.W_proj(x_conv_out))
        x_gate = self.sigmoid(self.W_out(x_conv_out))
        x_highway = x_gate * x_proj + (1-x_gate) * x_conv_out
        return x_highway
