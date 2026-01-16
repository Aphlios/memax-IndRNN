from beartype.typing import List, Optional, Tuple

import jax
import jax.numpy as jnp
from equinox import filter_vmap, nn
from jax import nn as jnn
from jaxtyping import Array, PRNGKeyArray, Shaped

from memax.equinox.groups import Module
from memax.mtypes import Input, ResetRecurrentState


def sinkhorn_log(log_alpha, num_iters=20, tau=0.05):
    log_alpha = log_alpha / tau

    def body(i, val):
        val = val - jnn.logsumexp(val, axis=-1, keepdims=True)
        val = val - jnn.logsumexp(val, axis=-2, keepdims=True)
        return val

    log_alpha = jax.lax.fori_loop(0, num_iters, body, log_alpha)
    return jnp.exp(log_alpha)


class MHCModel(Module):
    """Manifold Constrained Hyper-Connections (MHC) Model.
    
    Divides the recurrent state into multiple communicating streams.
    Based on: https://arxiv.org/abs/2512.24880
    """

    layers: List[Module]
    ff: List[nn.Sequential]
    
    h_res_logits: List[Array]
    h_pre_logits: List[Array]
    h_post_logits: List[Array]
    
    map_in: nn.Linear
    map_out: nn.Linear
    
    num_streams: int
    stream_dim: int
    mhc_num_iters: int
    mhc_tau: float

    def __init__(
        self,
        make_layer_fn,
        input_size,
        output_size,
        recurrent_size,
        num_streams=4,
        num_layers=2,
        mhc_num_iters=10,
        mhc_tau=0.05,
        activation=jax.nn.leaky_relu,
        *,
        key
    ):
        self.num_streams = num_streams
        assert recurrent_size % num_streams == 0, (
            f"recurrent_size ({recurrent_size}) must be divisible by "
            f"num_streams ({num_streams})"
        )
        self.stream_dim = recurrent_size // num_streams
        self.mhc_num_iters = mhc_num_iters
        self.mhc_tau = mhc_tau

        keys = jax.random.split(key, 3)
        self.map_in = nn.Linear(input_size, recurrent_size, key=keys[0])
        self.map_out = nn.Linear(recurrent_size, output_size, key=keys[1])
        
        self.layers = []
        self.ff = []
        self.h_res_logits = []
        self.h_pre_logits = []
        self.h_post_logits = []
        
        layer_key = keys[2]
        
        for _ in range(num_layers):
            layer_key, lk, ffk, pre_k = jax.random.split(layer_key, 4)
            
            # Create recurrent layer for a single stream
            self.layers.append(make_layer_fn(recurrent_size=self.stream_dim, key=lk))
            
            # Feed-forward block per layer
            self.ff.append(
                nn.Sequential(
                    [
                        nn.Linear(self.stream_dim, self.stream_dim, key=ffk),
                        nn.LayerNorm(
                            (self.stream_dim,), use_weight=False, use_bias=False
                        ),
                        nn.Lambda(activation),
                    ]
                )
            )
            
            # Logic:
            # -8 everywhere, 0 on diagonal for h_res (encourages identity initially)
            h_res = jnp.full((num_streams, num_streams), -8.0)
            h_res = h_res.at[jnp.diag_indices(num_streams)].set(0.0)
            self.h_res_logits.append(h_res)
            
            # Random stream index for initial branch input
            idx = jax.random.randint(pre_k, (), 0, num_streams)
            h_pre = jnp.full((1, num_streams), -8.0)
            h_pre = h_pre.at[0, idx].set(0.0)
            self.h_pre_logits.append(h_pre)
            
            # Zero initialization for branch output mixing
            h_post = jnp.zeros((1, num_streams))
            self.h_post_logits.append(h_post)

    def __call__(
        self, h: ResetRecurrentState, x: Input, key: Optional[PRNGKeyArray] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        emb, start = x
        
        # Map input to total hidden size
        emb = filter_vmap(self.map_in)(emb)
        
        T = emb.shape[0]
        # Reshape to (Time, Streams, StreamDim)
        residuals = emb.reshape(T, self.num_streams, self.stream_dim)
        
        h_out = []
        
        for i, (ff, recurrent_layer, h_i) in enumerate(zip(self.ff, self.layers, h)):
            if key is None:
                key, rkey = None, None
            else:
                key, rkey = jax.random.split(key)
            
            # 1. Width Connection (Inter-stream communication)
            h_res_mat = sinkhorn_log(
                self.h_res_logits[i], self.mhc_num_iters, self.mhc_tau
            )
            # Mix streams: residuals[t, out_s, d] = sum(h_res[in_s, out_s] * residuals[t, in_s, d])
            # aligned with PyTorch 's t, ... s d -> ... t d'
            residuals = jnp.einsum("st,tsd->ttd", h_res_mat, residuals)
            
            # 2. Extract Branch Input
            h_pre_w = jnn.softmax(self.h_pre_logits[i], axis=-1) # (1, S)
            # branch_in[t, 1, d] = sum(h_pre[1, s] * residuals[t, s, d])
            branch_in_seq = jnp.einsum("vs,tsd->tvd", h_pre_w, residuals)
            branch_in_seq = branch_in_seq.squeeze(1) # (T, D)
            
            # 3. Branch Execution (Recurrent + FF)
            tmp, branch_out_seq = recurrent_layer(h_i, (branch_in_seq, start), key=rkey)
            h_out.append(tmp)
            
            branch_out_seq = filter_vmap(ff)(branch_out_seq) # (T, D)
            
            # 4. Depth Connection (Merge back)
            h_post_w = jnn.softmax(self.h_post_logits[i], axis=-1) # (1, S)
            # update[t, s, d] = sum(h_post[1, s] * branch_out[t, 1, d])
            branch_out_expanded = branch_out_seq[:, None, :] # (T, 1, D)
            residuals_update = jnp.einsum("vs,tvd->tsd", h_post_w, branch_out_expanded)
            
            residuals = residuals + residuals_update
        
        # Flatten streams and project to output
        final_seq = residuals.reshape(T, -1)
        out = filter_vmap(self.map_out)(final_seq)
        
        return tuple(h_out), out

    def initialize_carry(
        self, key: Optional[Shaped[PRNGKeyArray, ""]] = None
    ) -> Tuple[ResetRecurrentState, ...]:
        if key is None:
            keys = tuple(None for _ in range(len(self.layers)))
        else:
            keys = jax.random.split(key, len(self.layers))
        return tuple(l.initialize_carry(k) for l, k in zip(self.layers, keys))

    def latest_recurrent_state(self, h: ResetRecurrentState) -> ResetRecurrentState:
        return tuple(l.latest_recurrent_state(h_i) for l, h_i in zip(self.layers, h))
