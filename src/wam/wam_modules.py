"""WAM (World-Action Model) modules for DeepSight.

Three components:
  1. ActionEncoder : future waypoints (N x 2) -> action embedding (D)
  2. FiLM          : (action_emb) -> (gamma, beta) for AdaLN modulation
  3. BEVDecoder    : static scene context + action -> reconstructed future BEV RGB

Design intent:
  BEVDecoder takes a current-observation context (B,16,16,D), while the
  demonstrated trajectory conditions both latent and RGB future prediction.
  Reconstruction alone does not prove that the model uses the action; that is
  tested separately with no-action, shuffled-action and intervention controls.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def scale_gradient(tensor: torch.Tensor, scale: float) -> torch.Tensor:
    """Return ``tensor`` unchanged while scaling its backward gradient.

    ``scale=0`` is equivalent to ``detach()`` and ``scale=1`` preserves the
    original graph.  Intermediate values are useful for letting dense RGB
    reconstruction losses reach a shared planner without overwhelming it.
    """
    if not 0.0 <= scale <= 1.0:
        raise ValueError(f"gradient scale must be in [0, 1], got {scale}")
    if scale == 0.0:
        return tensor.detach()
    if scale == 1.0:
        return tensor

    detached = tensor.detach()
    return detached + (tensor - detached) * scale


class ActionEncoder(nn.Module):
    """Encode future waypoints into a single action embedding.

    Args:
        num_waypoints: number of future waypoints (DeepSight uses 4).
        hidden_size:   LLM hidden size (2048 for Qwen2.5-VL-3B).
        waypoint_scale: divide raw waypoints by this before encoding.
                        Typical BEV waypoints are in metres (0-50 m range);
                        dividing by ~10 keeps the linear-layer input near [-5,5].
    """

    def __init__(self, num_waypoints: int = 4, hidden_size: int = 2048,
                 waypoint_scale: float = 10.0):
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_scale = waypoint_scale
        in_dim = num_waypoints * 2
        self.fc1 = nn.Linear(in_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # std=0.05: stronger init so action signal enters gradients faster.
        # With waypoint_scale=10 inputs are O(1); std=0.01 was too weak.
        nn.init.normal_(self.fc1.weight, std=0.05)
        nn.init.zeros_(self.fc1.bias)
        nn.init.normal_(self.fc2.weight, std=0.05)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, waypoints: torch.Tensor) -> torch.Tensor:
        # waypoints: (B, num_waypoints, 2) -> (B, hidden_size)
        b = waypoints.shape[0]
        x = waypoints.reshape(b, -1)
        # Waypoint DATA (not params) can contain NaN from missing/padded future
        # frames; zero them before the matmul. This is a legitimate input guard.
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalise magnitude (waypoints are in metres) so fc1 input is O(1).
        x = x / self.waypoint_scale
        x = torch.nn.functional.gelu(self.fc1(x))
        x = self.fc2(x)
        return x


class FiLM(nn.Module):
    """Produce per-channel (gamma, beta) from an embedding for AdaLN.

    Zero-init: identity modulation at start (gamma=1, beta=0).
    Preserves pretrained DeepSight representations on load.
    """

    def __init__(self, hidden_size: int = 2048):
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, emb: torch.Tensor):
        gamma, beta = self.proj(emb).chunk(2, dim=-1)  # each (B, D)
        return 1.0 + gamma, beta


class BEVDecoder(nn.Module):
    """Action-conditioned future-BEV rollout decoder.

    Context (B,16,16,D) represents the current observation. Action embeddings
    and learned frame embeddings condition each predicted horizon. The module
    permits action-sensitive rollouts, but matched controls are still required
    to establish that the learned predictor uses the correct action pairing.
    """

    def __init__(self, hidden_size: int = 2048, num_frames: int = 5, base_ch: int = 256):
        super().__init__()
        self.num_frames = num_frames
        # Normalise LLM hidden states before projection: their variance can be
        # very small in bf16, making Conv weights produce near-zero outputs that
        # InstanceNorm then divides by → NaN.
        self.context_norm = nn.LayerNorm(hidden_size)
        # learnable frame index embeddings (small init so they don't overpower action)
        self.frame_embed = nn.Parameter(torch.randn(num_frames, hidden_size) * 0.02)
        # per-frame FiLM: zero-init = identity at start
        self.film_proj = nn.Linear(hidden_size, hidden_size * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

        self.in_proj = nn.Conv2d(hidden_size, base_ch, kernel_size=1)
        # Xavier-style init: input has ~unit variance after context_norm,
        # so std=1/sqrt(hidden_size) keeps output variance ≈ 1, which
        # prevents InstanceNorm from dividing by near-zero variance.
        nn.init.normal_(self.in_proj.weight, std=1.0 / (hidden_size ** 0.5))
        nn.init.zeros_(self.in_proj.bias)

        def up_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.InstanceNorm2d(cout, affine=True, eps=1e-5),
                nn.GELU(),
            )

        self.up = nn.Sequential(
            up_block(base_ch, base_ch),            # 16->32
            up_block(base_ch, base_ch // 2),       # 32->64
            up_block(base_ch // 2, base_ch // 4),  # 64->128
            up_block(base_ch // 4, base_ch // 8),  # 128->256
        )
        self.out_conv = nn.Conv2d(base_ch // 8, 3, kernel_size=3, padding=1)

    def _film(self, cond: torch.Tensor):
        g, b = self.film_proj(cond).chunk(2, dim=-1)
        return 1.0 + g, b

    def forward(self, context: torch.Tensor, action_emb: torch.Tensor) -> torch.Tensor:
        """
        context   : (B, 16, 16, D)  current-observation context
        action_emb: (B, D)
        returns   : (B, T, 3, 256, 256)
        """
        # Normalise LLM hidden-state context to ~unit variance so the 1x1
        # in_proj (small variance in bf16) doesn't collapse toward zero.
        context = self.context_norm(context)

        frames = []
        for f in range(self.num_frames):
            cond = action_emb + self.frame_embed[f]
            gamma, beta = self._film(cond)                    # zero-init → identity
            c = context * gamma[:, None, None, :] + beta[:, None, None, :]
            x = self.in_proj(c.permute(0, 3, 1, 2))           # (B, base_ch, 16, 16)
            x = self.up(x)                                    # (B, base_ch/8, 256, 256)
            x = self.out_conv(x)                              # (B, 3, 256, 256)
            frames.append(x)
        return torch.stack(frames, dim=1)                     # (B, T, 3, 256, 256)


class OccDecoder(nn.Module):
    """Action-conditioned BEV occupancy decoder for Stage-2 planning.

    Same inputs as BEVDecoder: (context, action_emb), but outputs binary
    occupancy logits at 64×64 per frame rather than 256×256 RGB. Occupancy
    at 64×64 (~0.78 m/px for a 50 m BEV range) is sufficient for geometric
    collision checking while being ~16× cheaper than BEVDecoder.

    Supervised with BCEWithLogitsLoss against pseudo-labels extracted from
    CARLA's semantic BEV renders (road=gray/low-sat, agents=colored/high-sat).

    Output: raw logits (no sigmoid) — use BCEWithLogitsLoss for training and
    torch.sigmoid() for inference.
    """

    def __init__(self, hidden_size: int = 2048, num_frames: int = 5,
                 base_ch: int = 128):
        super().__init__()
        self.num_frames = num_frames
        self.context_norm = nn.LayerNorm(hidden_size)
        self.frame_embed = nn.Parameter(torch.randn(num_frames, hidden_size) * 0.02)
        # per-frame FiLM: zero-init = identity at start
        self.film_proj = nn.Linear(hidden_size, hidden_size * 2)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        # 1×1 projection: hidden_size -> base_ch, xavier init
        self.in_proj = nn.Conv2d(hidden_size, base_ch, kernel_size=1)
        nn.init.normal_(self.in_proj.weight, std=1.0 / (hidden_size ** 0.5))
        nn.init.zeros_(self.in_proj.bias)

        def up_block(cin: int, cout: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.Conv2d(cin, cout, kernel_size=3, padding=1),
                nn.InstanceNorm2d(cout, affine=True, eps=1e-5),
                nn.GELU(),
            )

        # Two upsampling blocks: 16→32→64
        self.up = nn.Sequential(
            up_block(base_ch, base_ch),       # 16 -> 32
            up_block(base_ch, base_ch // 2),  # 32 -> 64
        )
        # Single-channel output (occupancy logits); bias -2.0 ≈ 12% prior
        self.out_conv = nn.Conv2d(base_ch // 2, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.constant_(self.out_conv.bias, -2.0)

    def _film(self, cond: torch.Tensor):
        g, b = self.film_proj(cond).chunk(2, dim=-1)
        return 1.0 + g, b

    def forward(self, context: torch.Tensor,
                action_emb: torch.Tensor) -> torch.Tensor:
        """
        context   : (B, 16, 16, D)  current-observation context (no future info)
        action_emb: (B, D)
        returns   : (B, T, 1, 64, 64)  raw occupancy logits
        """
        context = self.context_norm(context)
        frames = []
        for f in range(self.num_frames):
            cond = action_emb + self.frame_embed[f]
            gamma, beta = self._film(cond)
            c = context * gamma[:, None, None, :] + beta[:, None, None, :]
            x = self.in_proj(c.permute(0, 3, 1, 2))  # (B, base_ch, 16, 16)
            x = self.up(x)                            # (B, base_ch//2, 64, 64)
            x = self.out_conv(x)                      # (B, 1, 64, 64)
            frames.append(x)
        return torch.stack(frames, dim=1)             # (B, T, 1, 64, 64)


def extract_bev_context(hidden_states: torch.Tensor,
                        bev_block_mask: torch.Tensor,
                        num_frames: int = 5,
                        tokens_per_frame: int = 261,
                        patch_grid: int = 16) -> torch.Tensor:
    """Extract BEV patch grid from hidden states and pool across frames.

    bev_block_mask: (B, L) bool, True for ALL bev-token positions (1305 total).
    Returns: context (B, 16, 16, D) — mean-pooled over frames, patch-grid only.

    NOTE: This function sources context from FUTURE BEV tokens and is kept only
    for reference. Use extract_obs_context() instead to avoid future-info leakage.
    """
    b = hidden_states.shape[0]
    d = hidden_states.shape[-1]
    # Gather all bev tokens: should be exactly B * num_frames * tokens_per_frame
    bev_all = hidden_states[bev_block_mask]  # (B * 1305, D)
    bev_per_sample = bev_all.reshape(b, num_frames, tokens_per_frame, d)
    # Drop the 5 register/CLS tokens (positions 0..4), keep 256 patch tokens
    patch_tokens = bev_per_sample[:, :, 5:, :]   # (B, T, 256, D)
    # Mean-pool across frames -> static scene context
    context = patch_tokens.mean(dim=1)            # (B, 256, D)
    context = context.reshape(b, patch_grid, patch_grid, d)
    return context


def extract_obs_context(hidden_states: torch.Tensor,
                        obs_mask: torch.Tensor,
                        patch_grid: int = 16) -> torch.Tensor:
    """Extract current-observation context from input image-pad token positions.

    Uses <|image_pad|> positions (wam_obs_masks from ad_collator) — these are
    the camera image tokens in the USER prompt. They contain ONLY current-frame
    information (no future target), avoiding the direct leakage caused when
    extract_bev_context() gave the decoder future-BEV hidden states. Correct
    use of the paired action still has to be established by matched controls.

    obs_mask  : (B, L) bool — True at every <|image_pad|> position.
    Returns   : context (B, patch_grid, patch_grid, D)

    The number of obs tokens varies with image resolution, so we adaptive-pool
    to exactly patch_grid² tokens before reshaping. This is robust regardless
    of the actual camera setup or resolution.
    """
    b, l, d = hidden_states.shape
    out = []
    for bi in range(b):
        toks = hidden_states[bi][obs_mask[bi]]   # (N_obs_i, D)
        if toks.shape[0] == 0:
            raise ValueError(
                f"sample {bi} has no current-observation image tokens; "
                "refusing a full-sequence fallback that could leak future targets"
            )
        # Adaptive pool: (1, D, N) -> (1, D, patch_grid²) -> (patch_grid², D)
        t = toks.T.unsqueeze(0)                          # (1, D, N)
        t = torch.nn.functional.adaptive_avg_pool1d(
            t.float(), patch_grid * patch_grid
        ).to(toks.dtype)                                  # (1, D, 256)
        out.append(t.squeeze(0).T)                        # (256, D)
    context = torch.stack(out, dim=0)                     # (B, 256, D)
    return context.reshape(b, patch_grid, patch_grid, d)  # (B, 16, 16, D)
