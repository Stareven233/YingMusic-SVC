# ===================== BASECFM Euler Visualizer (copy-paste) =====================
import os
from abc import ABC

import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm

from modules.diffusion_transformer import DiT
from utils.progress_bar import Dummy_Bar

dbar = Dummy_Bar()


@torch.no_grad()
def visualize_basecfm_euler_run(
    save_dir: str,
    traj: torch.Tensor,         # [S, B, C, T] 或 [S+1, B, C, T]
    prompt: torch.Tensor,       # [B, C, P] 或 [B, C, T]（仅前 P 为参考）
    y0: torch.Tensor,           # [B, C, T]  初始噪声（置零 prompt 段后的 x）
    snapshots: tuple = (0, -1), # 需要保存的 step 索引（支持负索引）
    sr_hint: int | None = None,
    hop_length_hint: int | None = None,
    only_first_batch: bool = True,
    max_frames: int | None = None,
):
    """
    保存：
      - prompt.png（参考段）
      - y0.png（初始）
      - traj_step_XXX.png（若干步）
      - traj_step_XXX_with_ref.png（把前导段替换为参考后的效果）
      - out_mel.png（最后一步）
    约定形状：mel 为 [B, C, T]；C=mel bins，T=帧。
    """
    # ---- 仅 rank0 落盘（DDP 安全） ----
    try:
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return
    except Exception:
        pass

    os.makedirs(save_dir, exist_ok=True)

    # ---- 统一搬到 CPU ----
    traj   = traj.detach().cpu()    # [S, B, C, T]
    prompt = prompt.detach().cpu()  # [B, C, P or T]
    y0     = y0.detach().cpu()      # [B, C, T]

    assert traj.dim() == 4, f"traj must be [S, B, C, T], got {traj.shape}"
    S, B, C, T = traj.shape
    b0 = 0 if only_first_batch else 0

    # ---- 可视化长度（横轴范围） ----
    Tvis = T if max_frames is None else min(T, int(max_frames))
    t0, t1, xlabel = 0.0, float(Tvis - 1), "Frames"
    if (sr_hint is not None) and (hop_length_hint is not None) and sr_hint > 0 and hop_length_hint > 0:
        hop_sec = hop_length_hint / float(sr_hint)
        t1 = hop_sec * (Tvis - 1)
        xlabel = "Time (s)"

    # ---- 稳健色域（统一 vmin/vmax，便于跨图对比） ----
    sub = traj[:, b0, :, :Tvis].reshape(-1)
    if sub.numel() > 0:
        vmin = float(sub.quantile(0.01).item())
        vmax = float(sub.quantile(0.99).item())
    else:
        vmin, vmax = None, None

    # ---- 画 [C, Tvis] 梅尔图：横轴=时间/帧，纵轴=mel-bin，带 colorbar ----
    def _save_mel(img_CT: torch.Tensor, path: str, title: str | None = None,
                  vmin_=None, vmax_=None):
        m = img_CT[:, :Tvis]  # [C, Tvis] on CPU
        plt.figure(figsize=(8, 4))
        im = plt.imshow(
            m, origin="lower", aspect="auto",
            extent=[t0, t1, 0, m.shape[0]-1],
            vmin=vmin_, vmax=vmax_, interpolation="nearest",
        )
        plt.xlabel(xlabel); plt.ylabel("Mel bin")
        if title: plt.title(title)
        cbar = plt.colorbar(im); cbar.set_label("Energy")
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    # ---- 取 batch 0 的 prompt / y0，并扩展 prompt 到 T ----
    prompt_in  = prompt[b0]       # [C, P] or [C, T]
    y00        = y0[b0]           # [C, T]
    prompt_full = torch.zeros(C, T, dtype=prompt_in.dtype)
    P = min(prompt_in.shape[-1], T)
    if P > 0:
        prompt_full[:, :P] = prompt_in[:, :P]

    # 前导布尔掩码：长度 = T，前 P 为 True
    mask0 = torch.zeros(T, dtype=torch.bool)
    if P > 0:
        mask0[:P] = True

    # ---- 保存 prompt / y0 ----
    _save_mel(prompt_full, os.path.join(save_dir, "prompt.png"),
              "prompt (reference segment)", vmin, vmax)
    _save_mel(y00, os.path.join(save_dir, "y0.png"),
              "y0 (initial)", vmin, vmax)

    # ---- 轨迹关键步 ----
    Smax = S - 1
    for si in snapshots:
        s = si if si >= 0 else (S + si)
        s = max(0, min(Smax, s))
        xt = traj[s, b0]  # [C, T]
        _save_mel(xt, os.path.join(save_dir, f"traj_step_{s:03d}.png"),
                  f"y_t at step {s}/{Smax}", vmin, vmax)

        # with ref：把前导 prompt 段替换为参考 mel
        xt_wr = xt.clone()
        xt_wr[:, mask0] = prompt_full[:, mask0]
        _save_mel(xt_wr, os.path.join(save_dir, f"traj_step_{s:03d}_with_ref.png"),
                  f"y_t (with ref) step {s}/{Smax}", vmin, vmax)

    # ---- 最终输出（last step） ----
    _save_mel(traj[-1, b0], os.path.join(save_dir, "out_mel.png"),
              "out (last step)", vmin, vmax)
# =================================================================



class BASECFM(torch.nn.Module, ABC):
    def __init__(
        self,
        args,
    ):
        super().__init__()
        self.sigma_min = 1e-6

        self.estimator = None

        self.in_channels = args.DiT.in_channels

        self.criterion = torch.nn.MSELoss() if args.reg_loss_type == "l2" else torch.nn.L1Loss()

        if hasattr(args.DiT, 'zero_prompt_speech_token'):
            self.zero_prompt_speech_token = args.DiT.zero_prompt_speech_token
        else:
            self.zero_prompt_speech_token = False

    @torch.inference_mode()
    def inference(self, mu, x_lens, prompt, style, f0, n_timesteps, temperature=1.0, inference_cfg_rate=0.5, style_r=None, pbar=tqdm):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """
        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate, style_r=style_r, pbar=pbar)

    def solve_euler(self, x, x_lens, prompt, mu, style, t_span, inference_cfg_rate=0.5, style_r=None, pbar=tqdm):
        """
        Fixed euler solver for ODEs.
        Args:
            x (torch.Tensor): random noise
            t_span (torch.Tensor): n_timesteps interpolated
                shape: (n_timesteps + 1,)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes
        """
        t, _, _ = t_span[0], t_span[-1], t_span[1] - t_span[0]

        # I am storing this because I can later plot it by putting a debugger here and saving it to a file
        # Or in future might add like a return_all_steps flag
        sol = []
        # apply prompt
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)    # torch.Size([1, 128, 1640])
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]  # torch.Size([1, 128, 349])
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        # ★ 记录 y0（置零后的初始）
        # y0_vis = x.clone()
        pbar = pbar or dbar


        for step in pbar(range(1, len(t_span))):
            dt = t_span[step] - t_span[step - 1]
            if inference_cfg_rate > 0:
                # Stack original and CFG (null) inputs for batched processing
                stacked_prompt_x = torch.cat([prompt_x, torch.zeros_like(prompt_x)], dim=0)       # prompt mel
                stacked_style = torch.cat([style, torch.zeros_like(style)], dim=0)                # timbre token
                stacked_mu = torch.cat([mu, torch.zeros_like(mu)], dim=0)                         # content
                stacked_x = torch.cat([x, x], dim=0)                                              # noise
                stacked_t = torch.cat([t.unsqueeze(0), t.unsqueeze(0)], dim=0)                    # time step
                if style_r is not None:
                    stacked_style_r = torch.cat([style_r, torch.zeros_like(style_r)], dim=0)
                else:
                    stacked_style_r=None

                # Perform a single forward pass for both original and CFG inputs
                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens, stacked_t, stacked_style, stacked_mu,style_r=stacked_style_r
                )

                # Split the output back into the original and CFG components
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)

                # Apply CFG formula
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
            else:
                dphi_dt = self.estimator(x, prompt_x, x_lens, t.unsqueeze(0), style, mu, style_r=style_r)


            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)

            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
            x[:, :, :prompt_len] = 0

        # # 在 solve_euler 的循环结束后：
        # x_out = sol[-1]
        #
        # # ★ 可视化（不影响原有逻辑）
        # try:
        #     traj = torch.stack(sol, dim=0)  # [S, B, C, T]
        #     visualize_basecfm_euler_run(
        #         save_dir="/user-fs/chenzihao/chengongyu/svc/f5svcp2/output/debug/euler_run_0001",
        #         traj=traj,
        #         prompt=prompt,  # 你传进来的 prompt: [B, C, P]
        #         y0=y0_vis,  # 你在进入循环前保存的 x.clone()
        #         snapshots=tuple(range(len(t_span))),
        #         # 如果你有采样率/帧移，可填上；没有就留空，坐标按帧
        #         sr_hint=getattr(getattr(self, 'mel_spec', None), 'target_sample_rate', None),
        #         hop_length_hint=getattr(getattr(self, 'mel_spec', None), 'hop_length', None),
        #         only_first_batch=True,
        #         max_frames=1200,
        #     )
        # except Exception as e:
        #     print(f"[Euler-Vis] skip visualization: {e}")

        return sol[-1]



    def forward_for_loss(self, x1, x_lens, prompt_lens, mu, style):
        b, _, t = x1.shape
        # random timestep
        t = torch.full([b, 1, 1], 0.5, device=mu.device, dtype=x1.dtype)
        # sample noise p(x_0)
        r_loss = 0
        roll = 3
        for _ in range(roll):
            z = torch.randn_like(x1)

            y = (1 - (1 - self.sigma_min) * t) * z + t * x1
            u = x1 - (1 - self.sigma_min) * z

            prompt = torch.zeros_like(x1)
            for bib in range(b):
                prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
                # range covered by prompt are set to 0
                y[bib, :, :prompt_lens[bib]] = 0
                if self.zero_prompt_speech_token:
                    mu[bib, :, :prompt_lens[bib]] = 0

            estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens)  # [B, D, T]
            loss = 0

            for bib in range(b):
                loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
            loss /= b
            r_loss+=loss
        r_loss /= roll
        return r_loss, estimator_out + (1 - self.sigma_min) * z

    def solve_euler_ckpt(self, x, x_lens, prompt, mu, style, t_span, inference_cfg_rate=0.5, use_ckpt=True):
        """
        Fixed euler solver for ODEs with optional activation checkpointing.
        """
        # 初始时间
        t = t_span[0]

        # ---- 准备 prompt 区域 ----
        prompt_len = prompt.size(-1)
        prompt_x = torch.zeros_like(x)
        prompt_x[..., :prompt_len] = prompt[..., :prompt_len]
        x[..., :prompt_len] = 0
        if self.zero_prompt_speech_token:
            mu[..., :prompt_len] = 0

        # ---- 把“算 dphi_dt”封到闭包里，便于 checkpoint ----
        # 只接收 Tensor 做入参（checkpoint 的要求），其它用闭包捕获
        def _step(x_cur, t_cur, prompt_x_cur, x_lens_cur, style_cur, mu_cur):
            if inference_cfg_rate > 0:
                # batched CFG
                stacked_prompt_x = torch.cat([prompt_x_cur, torch.zeros_like(prompt_x_cur)], dim=0)
                stacked_style = torch.cat([style_cur, torch.zeros_like(style_cur)], dim=0)
                stacked_mu = torch.cat([mu_cur, torch.zeros_like(mu_cur)], dim=0)
                stacked_x = torch.cat([x_cur, x_cur], dim=0)
                stacked_t = torch.cat([t_cur.unsqueeze(0), t_cur.unsqueeze(0)], dim=0)

                stacked_dphi_dt = self.estimator(
                    stacked_x, stacked_prompt_x, x_lens_cur, stacked_t, stacked_style, stacked_mu
                )
                dphi_dt, cfg_dphi_dt = stacked_dphi_dt.chunk(2, dim=0)
                dphi_dt = (1.0 + inference_cfg_rate) * dphi_dt - inference_cfg_rate * cfg_dphi_dt
                return dphi_dt
            else:
                return self.estimator(x_cur, prompt_x_cur, x_lens_cur, t_cur.unsqueeze(0), style_cur, mu_cur)

        # ---- 主循环：每一步用 checkpoint 包住 _step ----
        # 注：PyTorch 2.x 通常 use_reentrant=False 更稳定
        for step in range(1, len(t_span)):
            dt = t_span[step] - t

            if use_ckpt and self.training:
                dphi_dt = checkpoint(_step, x, t, prompt_x, x_lens, style, mu, use_reentrant=False)
            else:
                dphi_dt = _step(x, t, prompt_x, x_lens, style, mu)

            x = x + dt * dphi_dt
            t = t + dt

            # 维持 prompt 区域置零的约束
            if step < len(t_span) - 1:
                _ = t_span[step + 1] - t  # 你原来这行没用到，保留占位可删除
            x[:, :, :prompt_len] = 0

        return x

    def inference_for_GAN(self, mu, x_lens, prompt, style, n_timesteps=30, temperature=1.0, inference_cfg_rate=0.7):
        """Forward diffusion

        Args:
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): output_mask
                shape: (batch_size, 1, mel_timesteps)
            n_timesteps (int): number of diffusion steps
            temperature (float, optional): temperature for scaling noise. Defaults to 1.0.
            spks (torch.Tensor, optional): speaker ids. Defaults to None.
                shape: (batch_size, spk_emb_dim)
            cond: Not used but kept for future purposes

        Returns:
            sample: generated mel-spectrogram
                shape: (batch_size, n_feats, mel_timesteps)
        """

        B, T = mu.size(0), mu.size(1)
        z = torch.randn([B, self.in_channels, T], device=mu.device) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        # t_span = t_span + (-1) * (torch.cos(torch.pi / 2 * t_span) - 1 + t_span)
        return self.solve_euler_ckpt(z, x_lens, prompt, mu, style, t_span, inference_cfg_rate)

    def forward(self, x1, x_lens, prompt_lens, mu, style, generate_x1=False, style_r=None, balance_loss=False):
        """Computes diffusion loss

        Args:
            x1 (torch.Tensor): Target
                shape: (batch_size, n_feats, mel_timesteps)
            mask (torch.Tensor): target mask
                shape: (batch_size, 1, mel_timesteps)
            mu (torch.Tensor): output of encoder
                shape: (batch_size, n_feats, mel_timesteps)
            spks (torch.Tensor, optional): speaker embedding. Defaults to None.
                shape: (batch_size, spk_emb_dim)

        Returns:
            loss: conditional flow matching loss
            y: conditional flow
                shape: (batch_size, n_feats, mel_timesteps)
        """

        # 正常forward
        if not generate_x1 :
            b, _, t = x1.shape
            # random timestep
            t = torch.rand([b, 1, 1], device=mu.device, dtype=x1.dtype)
            # sample noise p(x_0)
            z = torch.randn_like(x1)

            y = (1 - (1 - self.sigma_min) * t) * z + t * x1
            u = x1 - (1 - self.sigma_min) * z

            prompt = torch.zeros_like(x1)
            for bib in range(b):
                prompt[bib, :, :prompt_lens[bib]] = x1[bib, :, :prompt_lens[bib]]
                # range covered by prompt are set to 0
                y[bib, :, :prompt_lens[bib]] = 0
                if self.zero_prompt_speech_token:
                    mu[bib, :, :prompt_lens[bib]] = 0

            estimator_out = self.estimator(y, prompt, x_lens, t.squeeze(1).squeeze(1), style, mu, prompt_lens, style_r=style_r)

            ############ new energy balance & t-aware loss #################
            if balance_loss:
                w_bc = self._build_freq_time_weights(
                    u, x_lens, prompt_lens, t,
                    eps=1e-6, inv_sigma_max=1e3, ema=0.95, hi_ratio=0.25, lam=0.4
                )
                loss = x1.new_tensor(0.0)
                valid_cnt = 0
                for bib in range(b):
                    pl = int(prompt_lens[bib].item());
                    xl = int(x_lens[bib].item())
                    if xl <= pl:
                        continue
                    # 取片段
                    pred = estimator_out[bib, :, pl:xl]  # [C,L]
                    target = u[bib, :, pl:xl]  # [C,L]
                    # 把 (C,1) -> (C,L) 并取 sqrt 权重
                    w_sqrt = w_bc[bib].sqrt().expand(-1, pred.size(-1))  # [C,L]

                    # 关键：两边同乘 sqrt(w)，其余保持不变
                    loss += self.criterion(pred * w_sqrt, target * w_sqrt)
                    valid_cnt += 1

                loss = loss / valid_cnt

            #################################################################
            else:
                loss = 0
                for bib in range(b):
                    loss += self.criterion(estimator_out[bib, :, prompt_lens[bib]:x_lens[bib]], u[bib, :, prompt_lens[bib]:x_lens[bib]])
                loss /= b
            return loss, estimator_out + (1 - self.sigma_min) * z

        elif generate_x1:
            # 批次大小和特征数
            B, n_feats = x1.size(0), x1.size(1)
            device = x1.device
            # 存每个样本生成的结果
            gen_list = []
            for b in range(B):
                # 当前样本的有效长度
                T_b = int(x_lens[b].item())
                # 提取 mu -> [1, T_b, 768]
                mu_b = mu[b, :T_b, :].unsqueeze(0)
                # 提取 prompt -> [1, 128, P_b]
                P_b = int(prompt_lens[b].item())
                prompt_b = x1[b, :, :P_b].unsqueeze(0)
                # 提取 style -> [1, 192]
                style_b = style[b, :].unsqueeze(0)
                # 拓展生成长度到完整sample
                mu_pb = torch.cat([mu[b, :P_b, :].unsqueeze(0), mu_b], dim=1)
                # 单样本生成 -> [1, 128, T_b]
                # out_b = self.inference_for_GAN(
                #     mu=mu_b,
                #     x_lens=torch.tensor([T_b], device=device),
                #     prompt=prompt_b,
                #     style=style_b,
                # )
                out_b = self.inference_for_GAN(
                    mu=mu_pb,
                    x_lens=torch.tensor([T_b+P_b], device=device),
                    prompt=prompt_b,
                    style=style_b,
                )
                # 去掉 batch 维度 -> [128, T_b]
                gen_list.append(out_b[:,:,P_b:].squeeze(0))

            ############################################
                # plot_mel(gen_list[b],save_path=f'/ailab-train/speech/chengongyu/seedsvc/seed-vc/test/debug/gen{b}.png')
                # plot_mel(x1[b].squeeze(0),
                #          save_path=f'/ailab-train/speech/chengongyu/seedsvc/seed-vc/test/debug/gt{b}.png')
                # import pdb
                # pdb.set_trace()
            ############################################


            # pad 回 batch 张量，形状 [B, 128, max_T]
            max_T = int(torch.max(x_lens).item())
            y_gen = x1.new_zeros((B, n_feats, max_T))
            for b, gen in enumerate(gen_list):
                y_gen[b, :, :gen.size(1)] = gen
            # 生成分支无 loss
            return y_gen

#################################################################################
    def _should_sync_stats(self) -> bool:
        # 优先 Accelerate
        if hasattr(self, "accelerator"):
            try:
                return getattr(self.accelerator, "num_processes", 1) > 1
            except Exception:
                pass
        return _is_dist_initialized() and _world_size() > 1

    # ===== 主函数：直接返回组合后的 w_bc（B, C, 1）=====
    @torch.no_grad()
    def _build_freq_time_weights(
            self,
            u: torch.Tensor,  # [B, C, T] 目标向量场(建议用 u)
            x_lens: torch.Tensor,  # [B]
            prompt_lens: torch.Tensor,  # [B]
            t_diff: torch.Tensor,  # [B,1,1] 扩散时间 t∈[0,1]
            *,
            eps: float = 1e-6,
            inv_sigma_max: float = 1e3,
            ema: float = 0.95,
            hi_ratio: float = 0.30,  # 提升的高频比例（最高 30%）
            lam: float = 0.6,  # 时间×高频增益强度
    ) -> torch.Tensor:
        """
        返回 w_bc (B, C, 1)，已做：
          - 仅监督区间统计每通道方差 => inv_sigma
          - EMA 平滑 + 多卡同步 + 均值归一
          - 高频增益 g(c) + 时间课表 s(t)（E[s]=1）
          - 样本级均值归一 + 上限裁剪
        """
        assert u.dim() == 3 and t_diff.dim() == 3
        B, C, T = u.shape
        device, dtype = u.device, u.dtype

        # --- 监督区间 mask: [B, T] ---
        mask_bt = torch.zeros(B, T, device=device, dtype=dtype)
        for b in range(B):
            pl = int(prompt_lens[b].item());
            xl = int(x_lens[b].item())
            if xl > pl:
                mask_bt[b, pl:xl] = 1.0

        # --- 带 mask 的 per-channel 方差 -> inv_sigma ---
        u_btc = u.transpose(1, 2).contiguous()  # [B,T,C]
        m_btc = mask_bt.unsqueeze(-1)  # [B,T,1]
        msum_c = m_btc.sum((0, 1)).clamp_min(1.0)  # [C]
        mu_c = (u_btc * m_btc).sum((0, 1)) / msum_c  # [C]
        var_c = ((u_btc - mu_c.view(1, 1, -1)) ** 2 * m_btc).sum((0, 1)) / msum_c
        sigma_c = torch.sqrt(var_c.clamp_min(eps))  # [C]
        inv_sigma = (1.0 / sigma_c).clamp(max=inv_sigma_max)  # [C]

        # --- EMA 平滑（持久化 buffer） ---
        _register_buffer_if_needed(self, "_inv_sigma_running", inv_sigma.detach().clone())
        if ema is not None:
            self._inv_sigma_running.mul_(ema).add_(inv_sigma.detach(), alpha=(1.0 - ema))
            inv_sigma = self._inv_sigma_running

        # --- 多卡同步（可选，自动判定） ---
        if self._should_sync_stats():
            _all_reduce_mean_(inv_sigma)

        # --- 全局均值归一，保持整体尺度稳定 ---
        inv_sigma = inv_sigma / inv_sigma.mean().clamp_min(1e-6)  # [C]

        # --- 高频增益 g(c) ---
        r = torch.linspace(0.0, 1.0, C, device=device, dtype=dtype)  # 0..1
        if hi_ratio > 0:
            cut = max(0, int((1.0 - hi_ratio) * C))
            g = torch.zeros(C, device=device, dtype=dtype)
            if cut < C:
                g[cut:] = torch.linspace(0.0, 1.0, C - cut, device=device, dtype=dtype)
        else:
            g = torch.zeros(C, device=device, dtype=dtype)

        # --- 时间课表 s(t)，E[s]=1（t~U[0,1], κ=2 => E[t^2]=1/3） ---
        s = t_diff.view(-1).pow(2.0) / (1.0 / 3.0)  # [B]

        # --- 组合 w_{b,c}(t) = inv_sigma_c * (1 + λ * s_b * g_c) ---
        w_bc = inv_sigma.view(1, C, 1) * (1.0 + lam * s.view(-1, 1, 1) * g.view(1, C, 1))
        # 上限 & 每样本均值归一（mean over C）
        w_bc = w_bc.clamp(max=1e4)
        w_bc = w_bc / w_bc.mean(dim=1, keepdim=True).clamp_min(1e-6)  # (B,C,1)

        return w_bc

def _register_buffer_if_needed(module, name, tensor):
    if not hasattr(module, name) or getattr(module, name) is None:
        module.register_buffer(name, tensor)

def _is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def _world_size() -> int:
    return dist.get_world_size() if _is_dist_initialized() else 1

@torch.no_grad()
def _all_reduce_mean_(x: torch.Tensor):
    if _is_dist_initialized() and _world_size() > 1:
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
        x /= _world_size()
    return x



def plot_mel(
    mel,
    sr: int | None = None,          # 采样率（用于把帧换算为秒，可选）
    hop_length: int | None = None,  # 帧移（用于时间轴，可选）
    save_path: str | None = None,   # 不传就 plt.show()，传了就保存
    time_first: bool = False,       # 若你的输入是 [T, mel]，设为 True 会自动转置
    title: str | None = None,
    vmin: float | None = None,      # 若是 dB，可以设置下限比如 -80
    vmax: float | None = None       # 若是 dB，可以设置上限比如 0
):
    import numpy as np

    # 转成 numpy
    if isinstance(mel, torch.Tensor):
        x = mel.detach().cpu().float().numpy()
    else:
        x = np.asarray(mel, dtype=np.float32)

    # 确保是 2D
    if x.ndim != 2:
        raise ValueError(f"Expected 2D tensor [mel, T] or [T, mel], got shape {x.shape}")

    # 若输入是 [T, mel]，转为 [mel, T]
    if time_first:
        x = x.T

    # 计算横轴坐标范围（按秒）
    extent = None
    xlabel = "Frame"
    if sr is not None and hop_length is not None:
        duration = x.shape[1] * hop_length / sr
        extent = [0.0, duration, 0.0, x.shape[0]]
        xlabel = "Time (s)"

    # 画图
    plt.figure(figsize=(10, 4))
    plt.imshow(x, aspect='auto', origin='lower', extent=extent, vmin=vmin, vmax=vmax)
    plt.xlabel(xlabel)
    plt.ylabel("Mel bin")
    if title is not None:
        plt.title(title)
    plt.colorbar(label="Amplitude / dB")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

class CFM(BASECFM):
    def __init__(self, args):
        super().__init__(
            args
        )
        if args.dit_type == "DiT":
            self.estimator = DiT(args)
        else:
            raise NotImplementedError(f"Unknown diffusion type {args.dit_type}")

##########################################################################################
