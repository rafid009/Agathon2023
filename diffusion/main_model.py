import numpy as np
import torch
import torch.nn as nn
from diffusion.diff_models import diff_SAITS

class CSDI_base(nn.Module):
    def __init__(self, target_dim, config, device, is_simple=False):
        super().__init__()
        self.device = device
        self.target_dim = target_dim

        self.emb_time_dim = config["model"]["timeemb"]
        self.emb_feature_dim = config["model"]["featureemb"]
        self.is_unconditional = config["model"]["is_unconditional"]
        self.target_strategy = config["model"]["target_strategy"]
        self.model_type = config["model"]["type"]
        self.is_simple = is_simple

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        if self.is_unconditional == False:
            self.emb_total_dim += 1  # for conditional mask
        self.embed_layer = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.emb_feature_dim
        )

        config_diff = config["diffusion"]
        config_diff["side_dim"] = self.emb_total_dim

        # input_dim = 1 if self.is_unconditional == True else 2
        # if config['model']['type'] == 'SAITS':
        self.is_saits = True
        self.diffmodel = diff_SAITS(
            diff_steps=config['diffusion']['num_steps'],
            n_layers=config['model']['n_layers'],
            d_time=config['model']['d_time'],
            d_feature=config['model']['n_feature'],
            d_model=config['model']['d_model'],
            d_inner=config['model']['d_inner'],
            n_head=config['model']['n_head'],
            d_k=config['model']['d_k'],
            d_v=config['model']['d_v'],
            dropout=config['model']['dropout'],
            diff_emb_dim=config['diffusion']['diffusion_embedding_dim'],
            diagonal_attention_mask=config['model']['diagonal_attention_mask'],
            is_simple=self.is_simple
        )
        # else:
        #     self.is_saits = False
        #     self.diffmodel = diff_CSDI(config_diff, input_dim)

        # parameters for diffusion models
        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask, for_pattern_mask=None):
        if for_pattern_mask is None:
            for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        
        for i in range(len(cond_mask)):
            
            bm_choice = np.random.rand()
            if bm_choice > 0.5:
                rate = 0.3
                length = len(cond_mask[i])
                start_idx = np.random.randint(0, length - int(length*rate))
                cond_mask[i][start_idx:(start_idx + int(length*rate))] = 0
            else:
                mask_choice = np.random.rand()
                if self.target_strategy == "mix" and mask_choice > 0.5:
                    cond_mask[i] = rand_mask[i]
                else:
                    cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1]
        
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        # print(f"time: {time_embed.shape} and feat: {feature_embed.shape}")
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        if self.is_unconditional == False:
            side_mask = cond_mask.unsqueeze(1)  # (B,1,K,L)
            side_info = torch.cat([side_info, side_mask], dim=1)
        return side_info

    def calc_loss_valid(
        self, observed_data, cond_mask, observed_mask, side_info, is_train
    ):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate loss for all t
            loss = self.calc_loss(
                observed_data, cond_mask, observed_mask, side_info, is_train, set_t=t
            )
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(
        self, observed_data, cond_mask, observed_mask, side_info, is_train, set_t=-1
    ):
        B, K, L = observed_data.shape
        if is_train != 1:  # for validation
            t = (torch.ones(B) * set_t).long().to(self.device)
        else:
            t = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[t]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)
        target_mask = observed_mask - cond_mask
        num_eval = target_mask.sum()
        if self.is_saits:
            temp_mask = cond_mask.unsqueeze(dim=1)
            if self.is_simple:
                inputs = {
                    'X': total_input,
                    'missing_mask': cond_mask
                }
            else:
                total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
                inputs = {
                    'X': total_input,
                    'missing_mask': total_mask
                }
            predicted_1, predicted_2, predicted_3 = self.diffmodel(inputs, t)
            residual_1 = (noise - predicted_1) * target_mask
            residual_2 = (noise - predicted_2) * target_mask
            residual_3 = (noise - predicted_3) * target_mask
            loss = ((residual_1 ** 2).sum() + (residual_2 ** 2).sum() + (residual_3 ** 2).sum()) / (3 * (num_eval if num_eval > 0 else 1))
        else:
            predicted = self.diffmodel(total_input, side_info, t)  # (B,K,L)
            residual = (noise - predicted) * target_mask
            loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        if self.is_unconditional == True:
            total_input = noisy_data.unsqueeze(1)  # (B,1,K,L)
        else:
            if self.is_simple:
                total_input = cond_mask * observed_data + (1 - cond_mask) * noisy_data
                # total_input = total_input.unsqueeze(1)
            else:
                cond_obs = (cond_mask * observed_data).unsqueeze(1)
                noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
                total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
            
        return total_input

    def impute(self, observed_data, cond_mask, side_info, n_samples):
        B, K, L = observed_data.shape
        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            # print(f"sample {i}")
            # generate noisy observation for unconditional model
            if self.is_unconditional == True:
                noisy_obs = observed_data
                noisy_cond_history = []
                for t in range(self.num_steps):
                    noise = torch.randn_like(noisy_obs)
                    noisy_obs = (self.alpha_hat[t] ** 0.5) * noisy_obs + self.beta[t] ** 0.5 * noise
                    noisy_cond_history.append(noisy_obs * cond_mask)

            current_sample = torch.randn_like(observed_data)
            ti = 0
            for t in range(self.num_steps - 1, -1, -1):
                # print(f"diff step: {ti}")
                if self.is_unconditional == True:
                    diff_input = cond_mask * noisy_cond_history[t] + (1.0 - cond_mask) * current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,K,L)
                else:
                    if self.is_simple:
                        diff_input = cond_mask * observed_data + (1 - cond_mask) * current_sample
                        # diff_input = diff_input.unsqueeze(1)
                    else:
                        # print(f"not simple")
                        cond_obs = (cond_mask * observed_data).unsqueeze(1)
                        noisy_target = ((1 - cond_mask) * current_sample).unsqueeze(1)
                        diff_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
                if self.is_saits:
                    # print(f"saits")
                    temp_mask = cond_mask.unsqueeze(dim=1)
                    if not self.is_simple:
                        # print(f"again not simple mask")
                        total_mask = torch.cat([temp_mask, (1 - temp_mask)], dim=1)
                    else:
                        total_mask = cond_mask
                    inputs = {
                        'X': diff_input,
                        'missing_mask': total_mask
                    }
                    _, _, predicted = self.diffmodel(inputs, torch.tensor([t]).to(self.device))
                    
                else:
                    predicted = self.diffmodel(diff_input, side_info, torch.tensor([t]).to(self.device))
                # print(f"{'SAITS' if self.is_saits else 'CSDI'} predicted: {predicted}")
                # print(f"{'SAITS' if self.is_saits else 'CSDI'} alpha hat [t]: {self.alpha_hat[t]}")
                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                # print(f"{'SAITS' if self.is_saits else 'CSDI'} alpha [t]: {self.alpha[t]}")
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                # print(f"{'SAITS' if self.is_saits else 'CSDI'} coeff1: {coeff1} and coeff2: {coeff2}")
                current_sample = coeff1 * (current_sample - coeff2 * predicted)
                # print(f"{'SAITS' if self.is_saits else 'CSDI'} current: {current_sample}")
                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    # print(f"{'SAITS' if self.is_saits else 'CSDI'} beta [t]: {self.beta[t]} and sigma: {sigma}\nnoise: {noise}")
                    current_sample += sigma * noise
                # print(f"in time step {ti} {'SAITS' if self.is_saits else 'CSDI'}: {current_sample}")
                ti += 1
            current_sample = (1 - cond_mask) * current_sample + cond_mask * observed_data
            # print(f"curr sample: {current_sample.shape}")
            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, batch, is_train=1):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            _, _, _
        ) = self.process_data(batch)
        if is_train == 0:
            cond_mask = gt_mask
        elif self.target_strategy != "random":
            cond_mask = self.get_hist_mask(
                observed_mask, for_pattern_mask=for_pattern_mask
            )
        else:
            cond_mask = self.get_randmask(observed_mask)
        # print(f"cond: {cond_mask.shape}")
        side_info = self.get_side_info(observed_tp, cond_mask)
        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid
        return loss_func(observed_data, cond_mask, observed_mask, side_info, is_train)

    def evaluate(self, batch, n_samples):
        (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            _,
            cut_length,
            obs_data_inact,
            gt_intact
        ) = self.process_data(batch)

        with torch.no_grad():
            cond_mask = gt_mask
            target_mask = observed_mask - cond_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            samples = self.impute(observed_data, cond_mask, side_info, n_samples)

            for i in range(len(cut_length)):  # to avoid double evaluation
                target_mask[i, ..., 0 : cut_length[i].item()] = 0
        return samples, observed_data, target_mask, observed_mask, observed_tp, obs_data_inact, gt_intact


class CSDI_PM25(CSDI_base):
    def __init__(self, config, device, target_dim=36):
        super(CSDI_PM25, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        cut_length = batch["cut_length"].to(self.device).long()
        for_pattern_mask = batch["hist_mask"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)
        for_pattern_mask = for_pattern_mask.permute(0, 2, 1)

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Physio(CSDI_base):
    def __init__(self, config, device, target_dim=35):
        super(CSDI_Physio, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
        )


class CSDI_Precipitation(CSDI_base):
    def __init__(self, config, device, target_dim=45):# 45 number of features
        super(CSDI_Precipitation, self).__init__(target_dim, config, device)

    def process_data(self, batch):
        observed_data = batch["observed_data"].to(self.device).float()
        observed_mask = batch["observed_mask"].to(self.device).float()
        observed_tp = batch["timepoints"].to(self.device).float()
        gt_mask = batch["gt_mask"].to(self.device).float()
        observed_data_intact = batch["obs_data_intact"].to(self.device).float()
        gt_intact = batch["gt_intact"]#.to(self.device).float()

        observed_data = observed_data.permute(0, 2, 1)
        observed_mask = observed_mask.permute(0, 2, 1)
        gt_mask = gt_mask.permute(0, 2, 1)

        cut_length = torch.zeros(len(observed_data)).long().to(self.device)
        for_pattern_mask = observed_mask

        return (
            observed_data,
            observed_mask,
            observed_tp,
            gt_mask,
            for_pattern_mask,
            cut_length,
            observed_data_intact,
            gt_intact
        )
