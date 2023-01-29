import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pypots.imputation.transformer import EncoderLayer, PositionalEncoding
from pypots.imputation import SAITS

def get_torch_trans(heads=8, layers=1, channels=64):
    encoder_layer = nn.TransformerEncoderLayer(
        d_model=channels, nhead=heads, dim_feedforward=64, activation="gelu"
    )
    return nn.TransformerEncoder(encoder_layer, num_layers=layers)


def Conv1d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv1d(in_channels, out_channels, kernel_size)
    nn.init.kaiming_normal_(layer.weight)
    return layer

def Conv2d_with_init(in_channels, out_channels, kernel_size):
    layer = nn.Conv2d(in_channels, out_channels, kernel_size, padding='same')
    nn.init.kaiming_normal_(layer.weight)
    return layer


class DiffusionEmbedding(nn.Module):
    def __init__(self, num_steps, embedding_dim=128, projection_dim=None):
        super().__init__()
        if projection_dim is None:
            projection_dim = embedding_dim
        self.register_buffer(
            "embedding",
            self._build_embedding(num_steps, embedding_dim / 2),
            persistent=False,
        )
        self.projection1 = nn.Linear(embedding_dim, projection_dim)
        self.projection2 = nn.Linear(projection_dim, projection_dim)

    def forward(self, diffusion_step):
        x = self.embedding[diffusion_step]
        x = self.projection1(x)
        x = F.silu(x)
        x = self.projection2(x)
        x = F.silu(x)
        return x

    def _build_embedding(self, num_steps, dim=64):
        steps = torch.arange(num_steps).unsqueeze(1)  # (T,1)
        frequencies = 10.0 ** (torch.arange(dim) / (dim - 1) * 4.0).unsqueeze(0)  # (1,dim)
        table = steps * frequencies  # (T,dim)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim=1)  # (T,dim*2)
        return table

class diff_CSDI(nn.Module):
    def __init__(self, config, inputdim=2):
        super().__init__()
        self.channels = config["channels"]

        self.diffusion_embedding = DiffusionEmbedding(
            num_steps=config["num_steps"],
            embedding_dim=config["diffusion_embedding_dim"],
        )

        self.input_projection = Conv1d_with_init(inputdim, self.channels, 1)
        self.output_projection1 = Conv1d_with_init(self.channels, self.channels, 1)
        self.output_projection2 = Conv1d_with_init(self.channels, 1, 1)
        nn.init.zeros_(self.output_projection2.weight)

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    side_dim=config["side_dim"],
                    channels=self.channels,
                    diffusion_embedding_dim=config["diffusion_embedding_dim"],
                    nheads=config["nheads"],
                )
                for _ in range(config["layers"])
            ]
        )

    def forward(self, x, cond_info, diffusion_step):
        B, inputdim, K, L = x.shape
        x = x.reshape(B, inputdim, K * L)
        x = self.input_projection(x)
        x = F.relu(x)
        x = x.reshape(B, self.channels, K, L)
        diffusion_emb = self.diffusion_embedding(diffusion_step)
        skip = []
        for layer in self.residual_layers:
            x, skip_connection = layer(x, cond_info, diffusion_emb)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / math.sqrt(len(self.residual_layers))
        x = x.reshape(B, self.channels, K * L)
        x = self.output_projection1(x)  # (B,channel,K*L)
        x = F.relu(x)
        x = self.output_projection2(x)  # (B,1,K*L)
        x = x.reshape(B, K, L)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, side_dim, channels, diffusion_embedding_dim, nheads):
        super().__init__()
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.cond_projection = Conv1d_with_init(side_dim, 2 * channels, 1)
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(channels, 2 * channels, 1)

        self.time_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)
        self.feature_layer = get_torch_trans(heads=nheads, layers=1, channels=channels)

    def forward_time(self, y, base_shape):
        B, channel, K, L = base_shape
        if L == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 2, 1, 3).reshape(B * K, channel, L)
        y = self.time_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, K, channel, L).permute(0, 2, 1, 3).reshape(B, channel, K * L)
        return y

    def forward_feature(self, y, base_shape):
        B, channel, K, L = base_shape
        if K == 1:
            return y
        y = y.reshape(B, channel, K, L).permute(0, 3, 1, 2).reshape(B * L, channel, K)
        y = self.feature_layer(y.permute(2, 0, 1)).permute(1, 2, 0)
        y = y.reshape(B, L, channel, K).permute(0, 2, 3, 1).reshape(B, channel, K * L)
        return y

    def forward(self, x, cond_info, diffusion_emb):
        B, channel, K, L = x.shape
        base_shape = x.shape
        x = x.reshape(B, channel, K * L)

        diffusion_emb = self.diffusion_projection(diffusion_emb).unsqueeze(-1)  # (B,channel,1)
        y = x + diffusion_emb

        y = self.forward_time(y, base_shape)
        y = self.forward_feature(y, base_shape)  # (B,channel,K*L)
        y = self.mid_projection(y)  # (B,2*channel,K*L)

        _, cond_dim, _, _ = cond_info.shape
        cond_info = cond_info.reshape(B, cond_dim, K * L)
        cond_info = self.cond_projection(cond_info)  # (B,2*channel,K*L)
        y = y + cond_info

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        y = self.output_projection(y)

        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(base_shape)
        residual = residual.reshape(base_shape)
        skip = skip.reshape(base_shape)
        return (x + residual) / math.sqrt(2.0), skip

class ResidualEncoderLayer(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()
        # self.enc_layer = EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
        #                  diagonal_attention_mask)

        self.enc_layer_X = EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        self.enc_layer_eps = EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)
        self.mid_projection = Conv1d_with_init(channels, 2*channels, 1)
        # self.output_projection = Conv1d_with_init(1, 4, 1)
        self.output_projection = Conv1d_with_init(1, 4, 1)
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_projection = Conv1d_with_init(2, channels, 1)
        # Transposed
        # self.pre_enc_layer = Conv1d_with_init(channels, 1, 1)
        # Parallel
        self.pre_enc_layer = Conv1d_with_init(channels, 2, 1)
        # self.pre_enc_layer_eps = 
        self.out_skip_proj = Conv1d_with_init(2, 1, 1)

    ## Transposed
    # def forward(self, x, diffusion_emb):
    #     B, channel, K, L = x.shape
    #     x_proj = torch.transpose(x, 2, 3)
    #     x_temp = x_proj.reshape(B, channel, K * L)
    #     x_proj = self.init_projection(x_temp)
    #     _, channel_out, _ = x_proj.shape
    #     # x_proj = x_proj.reshape(B, channel_out, K * L)
    #     diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
    #     y = x_proj + diff_proj
    #     y = self.mid_projection(y)
    #     gate, filter = torch.chunk(y, 2, dim=1)
    #     y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
    #     # y = y.reshape(B, channel_out, K, L)
    #     y = self.pre_enc_layer(y)
    #     y = y.reshape(B, L, K)
    #     y = torch.transpose(y, 1, 2)
    #     y, attn_weights = self.enc_layer(y)
    #     _, K3, L3 = y.shape
    #     y = y.reshape(B, 1, K3 * L3)
    #     y = self.output_projection(y)
    #     residual, skip = torch.chunk(y, 2, dim=1)
    #     x = x.reshape(B, channel, K3, L3)
    #     residual = residual.reshape(B, channel, K3, L3)
    #     skip = F.relu(self.out_skip_proj(skip))
    #     skip = skip.reshape(B, K3, L3)
        
    #     return (x + residual) / math.sqrt(2.0), skip, attn_weights

    
    def forward(self, x, diffusion_emb):
        B, channel, K, L = x.shape
        x_proj = torch.transpose(x, 2, 3)
        x_temp = x_proj.reshape(B, channel, K * L)
        x_proj = self.init_projection(x_temp)
        # _, channel_out, _ = x_proj.shape
        # x_proj = x_proj.reshape(B, channel_out, K * L)
        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        y = x_proj + diff_proj
        y = self.mid_projection(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        # y = y.reshape(B, channel_out, K, L)
        y = self.pre_enc_layer(y)
        _, channel_out, _ = y.shape
        y = y.reshape(B, channel_out, L, K)
        y = torch.transpose(y, 2, 3)
        slice_X, slice_eps = torch.chunk(y, 2, dim=1)
        
        y1 = slice_X.reshape(B, K, L)
        # y1 = torch.transpose(y1, 1, 2)
        y1, attn_weights_X = self.enc_layer_X(y1)

        y2 = slice_eps.reshape(B, K, L)
        # y2 = torch.transpose(y2, 1, 2)
        y2, attn_weights_eps = self.enc_layer_eps(y2)

        y = y1 + y2#torch.stack((y1, y2), dim=1)

        _, K3, L3 = y.shape
        y = y.reshape(B, 1, K3 * L3)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        x = x.reshape(B, channel, K3, L3)
        residual = residual.reshape(B, channel, K3, L3)
        skip = F.relu(self.out_skip_proj(skip))
        skip = skip.reshape(B, K3, L3)
        attn_weights = (attn_weights_X + attn_weights_eps) / 2
        return (x + residual) / math.sqrt(2.0), skip, attn_weights

class ResidualEncoderLayer_v2(nn.Module):
    def __init__(self, channels, d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout,
            diffusion_embedding_dim=128, diagonal_attention_mask=True) -> None:
        super().__init__()
        self.enc_layer = EncoderLayer(d_time, actual_d_feature, d_model, d_inner, n_head, d_k, d_v, dropout, 0,
                         diagonal_attention_mask)

       
        self.mid_projection = Conv1d_with_init(channels, 2 * channels, 1)
        self.output_projection = Conv1d_with_init(1, 2, 1)
        self.diffusion_projection = nn.Linear(diffusion_embedding_dim, channels)
        self.init_projection = Conv1d_with_init(1, channels, 1)
        self.pre_enc_layer = Conv1d_with_init(channels, 1, 1)
        self.out_skip_proj = Conv1d_with_init(1, 1, 1)

    def forward(self, x, diffusion_emb):
        B, K, L = x.shape
        x_temp = x.reshape(B, 1, K * L)
        x_proj = self.init_projection(x_temp)
        _, channel_out, _ = x_proj.shape
        # x_proj = x_proj.reshape(B, channel_out, K * L)
        diff_proj = self.diffusion_projection(diffusion_emb).unsqueeze(-1)
        # print(f"diff proj: {diff_proj}")
        y = x_proj + diff_proj
        y = self.mid_projection(y)
        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)  # (B,channel,K*L)
        # print(f"y gat * filter: {y}")
        # y = y.reshape(B, channel_out, K, L)
        y = self.pre_enc_layer(y)
        # print(f"y pre enc: {y}")
        # slice_X, slice_eps = torch.chunk(y, 2, dim=1)
        y = y.reshape(B, K, L)
        y, attn_weights = self.enc_layer(y)
        # print(f"y post enc: {y}")
        _, K3, L3 = y.shape
        y = y.reshape(B, 1, K3 * L3)
        y = self.output_projection(y)
        # print(f"y out proj: {y}")
        residual, skip = torch.chunk(y, 2, dim=1)
        # print(f"res: {residual}\nskip res: {skip}")
        x = x.reshape(B, K3, L3)
        residual = residual.reshape(B, K3, L3)
        skip = F.relu(self.out_skip_proj(skip))
        skip = skip.reshape(B, K3, L3)
        return (x + residual) / math.sqrt(2.0), skip, attn_weights

class diff_SAITS(nn.Module):
    def __init__(self, diff_steps, diff_emb_dim, n_layers, d_time, d_feature, d_model, d_inner, n_head, d_k, d_v,
            dropout, diagonal_attention_mask=True, is_simple=False):
        super().__init__()
        self.n_layers = n_layers
        actual_d_feature = d_feature * 2
        self.is_simple = is_simple
        # self.ORT_weight = ORT_weight
        # self.MIT_weight = MIT_weight
        if self.is_simple:
            self.layer_stack_for_first_block = nn.ModuleList([
                ResidualEncoderLayer_v2(channels=32, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                ResidualEncoderLayer_v2(channels=32, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
        else:
            self.layer_stack_for_first_block = nn.ModuleList([
                ResidualEncoderLayer(channels=32, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
            self.layer_stack_for_second_block = nn.ModuleList([
                ResidualEncoderLayer(channels=32, d_time=d_time, actual_d_feature=actual_d_feature, 
                            d_model=d_model, d_inner=d_inner, n_head=n_head, d_k=d_k, d_v=d_v, dropout=dropout,
                            diffusion_embedding_dim=diff_emb_dim, diagonal_attention_mask=diagonal_attention_mask)
                for _ in range(n_layers)
            ])
        self.diffusion_embedding = DiffusionEmbedding(diff_steps, diff_emb_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.position_enc_x = PositionalEncoding(d_model, n_position=d_time)
        self.position_enc_mask = PositionalEncoding(d_model, n_position=d_time)
        # for operation on time dim
        self.embedding_1 = nn.Linear(actual_d_feature, d_model)
        self.reduce_dim_z = nn.Linear(d_model, d_feature)
        # for operation on measurement dim
        self.embedding_2 = nn.Linear(actual_d_feature, d_model)
        self.reduce_skip_z = nn.Linear(d_model, d_feature)
        self.reduce_dim_beta = nn.Linear(d_model, d_feature)
        self.reduce_dim_gamma = nn.Linear(d_feature, d_feature)
        # for delta decay factor
        self.weight_combine = nn.Linear(d_feature + d_time, d_feature)

    def forward(self, inputs, diffusion_step):
        # print(f"Entered forward")
        X, masks = inputs['X'], inputs['missing_mask']
        if self.is_simple:
            X = torch.transpose(X, 1, 2)
            masks = torch.transpose(masks, 1, 2)
        else:    
            X = torch.transpose(X, 2, 3)
            masks = torch.transpose(masks, 2, 3)
        # print(f"X: {X.shape}, masks: {masks.shape}")
        diff_emb = self.diffusion_embedding(diffusion_step)
        # first DMSA block
        if self.is_simple:
            input_X_for_first = torch.cat([X, masks], dim=2)
            input_X_for_first = self.embedding_1(input_X_for_first)
            enc_output = self.dropout(self.position_enc_x(input_X_for_first))
        else:
            input_X_for_first = torch.cat([X, masks], dim=3)
            input_X_for_first = self.embedding_1(input_X_for_first)
            enc_output_x = self.dropout(self.position_enc_x(input_X_for_first[:, 0, :, :]))  # namely, term e in the math equation
            enc_output_mask = self.dropout(self.position_enc_mask(input_X_for_first[:, 1, :, :]))
            enc_output_x = enc_output_x.unsqueeze(1)
            enc_output_mask = enc_output_mask.unsqueeze(1)
            enc_output = torch.cat([enc_output_x, enc_output_mask], dim=1)
            # print(f"tilde 1 enc_out before attn: {enc_output}")
        skips_tilde_1 = []
        for encoder_layer in self.layer_stack_for_first_block:
            enc_output, skip, _ = encoder_layer(enc_output, diff_emb)
            # print(f"enc out after first encoder: {enc_output}")
            # print(f"after first block each iter: {skip}")
            skips_tilde_1.append(skip)

        X_tilde_1 = self.reduce_dim_z(enc_output)
        skips_tilde_1 = torch.sum(torch.stack(skips_tilde_1), dim=0) / math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_1 = self.reduce_skip_z(skips_tilde_1)
        # print(f"skip tilde 1: {skips_tilde_1.shape}")
        X_tilde_1[:, 0, :, :] = masks[:, 0, :, :] * X[:, 0, :, :] + (1 - masks[:, 0, :, :]) * X_tilde_1[:, 0, :, :]
        X_tilde_1[:, 1, :, :] = masks[:, 1, :, :] * X_tilde_1[:, 1, :, :]
        # print(f"X_tilde 1: {X_tilde_1}")
        # print(f"skip tilde 1: {skips_tilde_1}")
        # second DMSA block
        if self.is_simple:
            input_X_for_second = torch.cat([X_tilde_1, masks], dim=2)
            input_X_for_second = self.embedding_2(input_X_for_second)
            enc_output = self.position_enc_x(input_X_for_second)
        else:
            input_X_for_second = torch.cat([X_tilde_1, masks], dim=3)
            input_X_for_second = self.embedding_2(input_X_for_second)
            enc_output_x = self.position_enc_x(input_X_for_second[:, 0, :, :])
            enc_output_mask = self.position_enc_mask(input_X_for_second[:, 1, :, :])
            enc_output_x = enc_output_x.unsqueeze(1)
            enc_output_mask = enc_output_mask.unsqueeze(1)
            enc_output = torch.cat([enc_output_x, enc_output_mask], dim=1)
            # print(f"tilde 2 enc_out before attn: {enc_output}")
        skips_tilde_2 = []
        for encoder_layer in self.layer_stack_for_second_block:
            enc_output, skip, attn_weights = encoder_layer(enc_output, diff_emb)
            skips_tilde_2.append(skip)
            # print(f"enc out after first encoder: {enc_output}")
            # print(f"after first block each iter: {skip}")

        skips_tilde_2 = torch.sum(torch.stack(skips_tilde_2), dim=0) / math.sqrt(len(self.layer_stack_for_first_block))
        skips_tilde_2 = self.reduce_dim_gamma(F.relu(self.reduce_dim_beta(skips_tilde_2)))

        # print(f"skip tilde 1: {skips_tilde_1}")
        # attention-weighted combine
        attn_weights = attn_weights.squeeze(dim=1)  # namely term A_hat in Eq.
        if len(attn_weights.shape) == 4:
            # if having more than 1 head, then average attention weights from all heads
            attn_weights = torch.transpose(attn_weights, 1, 3)
            attn_weights = attn_weights.mean(dim=3)
            attn_weights = torch.transpose(attn_weights, 1, 2)
        if self.is_simple:
            combining_weights = torch.sigmoid(
                self.weight_combine(torch.cat([masks, attn_weights], dim=2))
            )
        else:
            combining_weights = torch.sigmoid(
                self.weight_combine(torch.cat([masks[:, 0, :, :], attn_weights], dim=2))
            )  # namely term eta
        # combine X_tilde_1 and X_tilde_2
        skips_tilde_3 = (1 - combining_weights) * skips_tilde_2 + combining_weights * skips_tilde_1
        # print(f"skip tilde 3: {skips_tilde_3}")
        skips_tilde_1 = torch.transpose(skips_tilde_1, 1, 2)
        skips_tilde_2 = torch.transpose(skips_tilde_2, 1, 2)
        skips_tilde_3 = torch.transpose(skips_tilde_3, 1, 2)
        # X_c = masks * X + (1 - masks) * X_tilde_3  # replace non-missing part with original data
        return skips_tilde_1, skips_tilde_2, skips_tilde_3

