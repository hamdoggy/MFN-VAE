# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Sigmoid, Module
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import torch.nn.functional as F


class LayerNorm(Module):
    """
    channels_last (default)： (batch_size, height, width, channels)
    channels_first： (batch_size, channels, height, width)
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Encoder(Module):
    def __init__(self, inputs=256):
        super(Encoder, self).__init__()
        self.hidden1 = nn.Linear(inputs, 256)
        self.bn1 = nn.LayerNorm(256)
        self.relu = nn.GELU()

        self.hidden2 = nn.Linear(256, 128)
        self.bn2 = nn.LayerNorm(128)

        self.hidden3 = nn.Linear(128, 128)
        self.bn3 = nn.LayerNorm(128)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.relu(x)

        return x


class Decoder(Module):
    def __init__(self, inputs=96, output=256):
        super(Decoder, self).__init__()
        self.hidden1 = nn.Linear(inputs, 128)
        self.bn1 = nn.LayerNorm(128)
        self.relu = nn.ReLU()

        self.hidden2 = nn.Linear(128, 256)
        self.bn2 = nn.LayerNorm(256)

        self.hidden3 = nn.Linear(256, 256)
        self.bn3 = nn.LayerNorm(256)

        self.hidden4 = nn.Linear(256, output)
        self.bn4 = nn.LayerNorm(output)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.hidden3(x)
        x = self.bn3(x)
        x = self.relu(x)

        x = self.hidden4(x)
        x = self.bn4(x)
        x = self.tanh(x)

        return x


class Predictor(Module):
    def __init__(self, inputs=96):
        super(Predictor, self).__init__()
        self.hidden1 = nn.Linear(inputs, 10)
        self.bn1 = nn.LayerNorm(10)
        self.relu1 = nn.ReLU()
        nn.init.kaiming_normal_(self.hidden1.weight, nonlinearity='relu')

        self.hidden2 = nn.Linear(10, 8)
        self.bn2 = nn.LayerNorm(8)
        self.relu2 = nn.ReLU()
        nn.init.kaiming_normal_(self.hidden2.weight, nonlinearity='relu')

        self.hidden3 = Linear(8, 1)
        nn.init.kaiming_normal_(self.hidden3.weight, nonlinearity='relu')
        self.act = Sigmoid()

    def forward(self, x):
        x = self.hidden1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.hidden2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.hidden3(x)
        x = self.act(x)

        return x


class MLP(Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.encoder_t1wi = Encoder(inputs=n_inputs)
        self.encoder_flair = Encoder(inputs=n_inputs)
        self.encoder_dwi = Encoder(inputs=n_inputs)

        self.liner_t1wi_mean = nn.Linear(128, 128)
        self.liner_t1wi_var = nn.Linear(128, 128)
        self.liner_flair_mean = nn.Linear(128, 128)
        self.liner_flair_var = nn.Linear(128, 128)
        self.liner_dwi_mean = nn.Linear(128, 128)
        self.liner_dwi_var = nn.Linear(128, 128)

        self.decoder_t1wi = Decoder(inputs=128, output=n_inputs)
        self.decoder_flair = Decoder(inputs=128, output=n_inputs)
        self.decoder_dwi = Decoder(inputs=128, output=n_inputs)

        self.predictor = Predictor(inputs=128 * 3)

    def forward(self, x1, x2, x3):
        x1 = self.encoder_t1wi(x1)
        x2 = self.encoder_flair(x2)
        x3 = self.encoder_dwi(x3)

        # torch.chunk(a, 3, dim=1)

        t1wi_mean = self.liner_t1wi_mean(x1)
        t1wi_var = self.liner_t1wi_var(x1)
        flair_mean = self.liner_flair_mean(x2)
        flair_var = self.liner_flair_var(x2)
        dwi_mean = self.liner_dwi_mean(x3)
        dwi_var = self.liner_dwi_var(x3)

        output_decoder_t1wi = self.decoder_t1wi(x1)
        output_decoder_flair = self.decoder_flair(x2)
        output_decoder_dwi = self.decoder_dwi(x3)

        # print(logvar)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.predictor(x)

        return output_decoder_t1wi, output_decoder_flair, output_decoder_dwi, x, \
               t1wi_mean, t1wi_var, flair_mean, flair_var, dwi_mean, dwi_var


class MLP1(Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.encoder_t1wi = Encoder(inputs=n_inputs)
        self.encoder_flair = Encoder(inputs=n_inputs)
        self.encoder_dwi = Encoder(inputs=n_inputs)

        self.decoder_t1wi = Decoder(inputs=128, output=n_inputs)
        self.decoder_flair = Decoder(inputs=128, output=n_inputs)
        self.decoder_dwi = Decoder(inputs=128, output=n_inputs)

        self.predictor = Predictor(inputs=128 * 3)

    def forward(self, x1, x2, x3):
        x1 = self.encoder_t1wi(x1)
        x2 = self.encoder_flair(x2)
        x3 = self.encoder_dwi(x3)

        output_decoder_t1wi = self.decoder_t1wi(x1)
        output_decoder_flair = self.decoder_flair(x2)
        output_decoder_dwi = self.decoder_dwi(x3)

        # print(logvar)

        x = torch.cat((x1, x2, x3), dim=1)
        x = self.predictor(x)

        return output_decoder_t1wi, output_decoder_flair, output_decoder_dwi, x,


class MLP2(Module):
    def __init__(self, n_inputs):
        super().__init__()
        self.encoder_t1wi = Encoder(inputs=n_inputs)
        self.encoder_flair = Encoder(inputs=n_inputs)

        self.liner_t1wi_mean = nn.Linear(128, 128)
        self.liner_t1wi_var = nn.Linear(128, 128)
        self.liner_flair_mean = nn.Linear(128, 128)
        self.liner_flair_var = nn.Linear(128, 128)

        self.decoder_t1wi = Decoder(inputs=128, output=n_inputs)
        self.decoder_flair = Decoder(inputs=128, output=n_inputs)

        self.predictor = Predictor(inputs=128 * 2)

    def forward(self, x1, x2):
        x1 = self.encoder_t1wi(x1)
        x2 = self.encoder_flair(x2)

        t1wi_mean = self.liner_t1wi_mean(x1)
        t1wi_var = self.liner_t1wi_var(x1)
        flair_mean = self.liner_flair_mean(x2)
        flair_var = self.liner_flair_var(x2)

        output_decoder_t1wi = self.decoder_t1wi(x1)
        output_decoder_flair = self.decoder_flair(x2)

        # print(logvar)
        x = torch.cat((x1, x2), dim=1)
        x = self.predictor(x)

        return output_decoder_t1wi, output_decoder_flair, x, \
               t1wi_mean, t1wi_var, flair_mean, flair_var,
