########################################################################################################################
# Project: blind unmixing using DIP techniques.

# In this file, we provide the DIP module for Nonlinear coefficient estimation.
########################################################################################################################

import torch
import torch.nn as nn
from Code.Modules.ADIP import conv_block_torch


class Gamma_DIP_torch(nn.Module):
    def __init__(
        self,
        input_shape,
        output_shape,
        num_filters_up,
        num_filters_down,
        filter_size_up,
        filter_size_down,
        filter_size_skip,
        downsample_modes,
        strides,
        activation,
        padding,
        use_skip_module=False,
    ):
        super(Gamma_DIP_torch, self).__init__()

        """
        build DIP for Nonlinear coefficient estimation.
        :param input_shape: network input_shape=(num_endmembers**2, img_row, img_col)
        :param output_shape: network output_shape=(num_endmembers**2, img_row, img_col)
        :param num_filters_up: number of filters in upsampling
        :param num_filters_down: number of filters in downsampling
        :param filter_size_up: size of filters in upsampling
        :param filter_size_down: size of filters in downsampling
        :param filter_size_skip: size of filters in skip-connection
        :param downsample_modes: mode for downsampling, if mode='stride', use conv stride for downsampling
        :param strides: strides in dowmsampling
        :param upsample_mode: mode for upsampling
        :param activation: activation function in conv blocks
        :param padding: padding method used in conv op
        :param use_skip_module: if true, a skip module is used in the skip connection, otherwise, just a direct path.
        :return: torch model.
        """
        self.down_layers = nn.ModuleList()
        in_channel = None
        out_channel = input_shape[0]
        # downsampling
        for id, (n, k, s, downsample_mode) in enumerate(
            zip(num_filters_down, filter_size_down, strides, downsample_modes)
        ):
            in_channel = out_channel
            out_channel = n
            self.down_layers = conv_block_torch(
                self.down_layers,
                in_channel=in_channel,
                filters=out_channel,
                kernel_size=k,
                strides=s,
                pad=padding,
                downsample_mode=downsample_mode,
                activation=activation,
            )

        # upsampling
        self.up_layers = nn.ModuleList()
        for id, (n, k) in enumerate(zip(num_filters_up, filter_size_up)):
            in_channel = out_channel
            out_channel = n
            self.up_layers = conv_block_torch(
                self.up_layers,
                in_channel=in_channel,
                filters=out_channel,
                kernel_size=k,
                strides=1,
                pad=padding,
                downsample_mode="stride",
                activation=activation,
            )

        # skip connection
        self.use_skip_module = use_skip_module
        if self.use_skip_module:
            self.skip_layers = nn.ModuleList()
            in_channel = input_shape[0]
            out_channel = input_shape[0]
            self.skip_layers = conv_block_torch(
                self.skip_layers,
                in_channel=in_channel,
                filters=out_channel,
                kernel_size=filter_size_skip,
                strides=1,
                pad=padding,
                downsample_mode="stride",
                activation=activation,
            )
        # output
        self.out_layers = nn.ModuleList()
        in_channel = input_shape[0] * 2
        out_channel = output_shape[0]
        self.out_layers = conv_block_torch(
            self.out_layers,
            in_channel=in_channel,
            filters=out_channel,
            kernel_size=1,
            strides=1,
            pad=padding,
            downsample_mode="stride",
            activation="",
        )
        self.out_layers.append(nn.Softmax(dim=1))

    def forward(self, x):
        # down sample
        o = x
        for layer in self.down_layers:
            o = layer(o)

        # up sample
        for layer in self.up_layers:
            o = layer(o)

        # skip connection
        skip = x
        if self.use_skip_module:
            for layer in self.skip_layers:
                skip = layer(skip)
        # o = skip + o
        o = torch.cat((skip, o), dim=1)

        # output layer
        for layer in self.out_layers:
            o = layer(o)

        return o
