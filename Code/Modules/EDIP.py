########################################################################################################################
# Project: blind unmixing using DIP techniques.
# In this file, we provide the DIP module for endmember estimation.
########################################################################################################################


import torch
import torch.nn as nn
import torch.nn.functional as F

from Code.Modules.Losses import angle_distance_loss_torch


def conv_block_torch(
    layers, in_channel, filters, kernel_size, strides, pad, downsample_mode, activation
):
    # block consists of: conv1D+BN+Activation
    if downsample_mode == "stride":
        layers.append(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=strides,
                padding=pad,
            )
        )
    else:
        layers.append(
            nn.Conv1d(
                in_channels=in_channel,
                out_channels=filters,
                kernel_size=kernel_size,
                stride=1,
                padding=pad,
            )
        )
        layers.append(nn.AvgPool1d(kernel_size=strides, stride=strides))

    layers.append(nn.BatchNorm1d(num_features=filters))

    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(negative_slope=0.1))

    return layers


class EDIP_torch(nn.Module):
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
        super(EDIP_torch, self).__init__()
        """
        build DIP for endmember estimation.
        :param input_shape: network input_shape=(num_bands, num_endmembers)
        :param output_shape: network output_shape=(num_bands, num_endmembers)
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
        out_channel = input_shape[-2]
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
            in_channel = input_shape[-2]
            out_channel = input_shape[-2]
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
        in_channel = input_shape[-2]
        out_channel = output_shape[-2]
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
        self.out_layers.append(nn.Sigmoid())

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
        o = skip + o
        # o = torch.cat((skip, o), dim=1)

        # output layer
        for layer in self.out_layers:
            o = layer(o)

        return o


def EDIP_loss_torch(abudance, hsi_data):
    """
    :param abudance: shape=(1, num_endm, img_row, img_col)
    :param hsi_data: shape=(1, num_bands, img_row, img_col)
    :return:
    """

    def my_loss(y_true, y_pred):
        """

        :param y_true:
        :param y_pred: shape=(1,num_bands,num_endm)
        :return: mse
        """
        _, num_endm, img_row, img_col = abudance.shape
        abu = abudance[0]

        est_hsi_data = torch.tensordot(y_pred, abu, dims=1)
        mse = F.mse_loss(est_hsi_data, hsi_data)

        return mse

    return my_loss


def EDIP_angle_loss_torch(abudance, hsi_data):
    """
    :param abudance: shape=(1, num_endm, img_row, img_col)
    :param hsi_data: shape=(1, num_bands, img_row, img_col)
    :return:
    """

    angle_distance_loss = angle_distance_loss_torch()

    def my_loss(y_true, y_pred):
        """

        :param y_true:
        :param y_pred: shape=(1,num_bands,num_endm)
        :return: mse
        """
        _, num_endm, img_row, img_col = abudance.shape
        batch_size, num_bands, _ = y_pred.shape

        abu = abudance[0]

        est_hsi_data = torch.tensordot(y_pred, abu, dims=1)

        hsi = torch.reshape(hsi_data, shape=(1, num_bands, img_row * img_col))
        hsi = torch.transpose(hsi, 1, 2)

        est = torch.reshape(est_hsi_data, shape=(1, num_bands, img_row * img_col))
        est = torch.transpose(est, 1, 2)

        angle_loss = angle_distance_loss(hsi, est)
        return angle_loss

    return my_loss
