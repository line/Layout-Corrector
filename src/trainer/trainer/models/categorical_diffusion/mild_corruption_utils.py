"""
This file is derived from the following code:
https://github.com/microsoft/LayoutGeneration/blob/main/LayoutDiffusion/improved-diffusion/improved_diffusion/discrete_diffusion.py

Author: Junyi42
License: MIT License (https://github.com/microsoft/LayoutGeneration/blob/main/LICENSE)

Modifications have been made to the original file to fit the requirements of this project.

-------------------------------------------------------------------------------

Util funcs for Mild Forward Corruption Discrete Diffusion (LayoutDiffusion, ICCV2023)
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.to(t.device).gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def sum_except_batch(x, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def gaussian_matrix2(t, bt, num_bin = 128):
    r"""Computes transition matrix for q(x_t|x_{t-1}).

    This method constructs a transition matrix Q with
    decaying entries as a function of how far off diagonal the entry is.
    Normalization option 1:
    Q_{ij} =  ~ softmax(-val^2/beta_t)   if |i-j| <= transition_bands
             1 - \sum_{l \neq i} Q_{il}  if i==j.
             0                          else.

    Normalization option 2:
    tilde{Q}_{ij} =  softmax(-val^2/beta_t)   if |i-j| <= transition_bands
                     0                        else.

    Q_{ij} =  tilde{Q}_{ij} / sum_l{tilde{Q}_{lj}}

    Args:
      t: timestep. integer scalar (or numpy array?)

    Returns:
      Q_t: transition matrix. shape = (num_pixel_vals, num_pixel_vals).
    """
    num_pixel_vals = num_bin
    transition_bands = num_pixel_vals - 1

    beta_t = bt.numpy()[t]

    mat = np.zeros((num_pixel_vals, num_pixel_vals),
                    dtype=np.float64)

    # Make the values correspond to a similar type of gaussian as in the
    # gaussian diffusion case for continuous state spaces.
    values = np.linspace(start=0., stop=float(num_pixel_vals - 1), num=num_pixel_vals,
                          endpoint=True, dtype=np.float64)
    values = values * 2./ (num_pixel_vals - 1.)
    values = values[:transition_bands+1]
    values = -values * values / beta_t

    values = np.concatenate([values[:0:-1], values], axis=0)
    values = np.exp(values)/np.sum(np.exp(values),axis=0)
    values = values[transition_bands:]
    for k in range(1, transition_bands + 1):
      off_diag = np.full(shape=(num_pixel_vals - k,),
                          fill_value=values[k],
                          dtype=np.float64)

      mat += np.diag(off_diag, k=k)
      mat += np.diag(off_diag, k=-k)

    # Add diagonal values such that rows and columns sum to one.
    # Technically only the ROWS need to sum to one
    # NOTE: this normalization leads to a doubly stochastic matrix,
    # which is necessary if we want to have a uniform stationary distribution.
    diag = 1. - mat.sum(1)
    mat += np.diag(diag, k=0)
    return mat


def alpha_schedule(
    time_step,
    T_tilde,
    type_classes,
    pos_classes = 128,
) -> Dict[str, np.ndarray]:
    # TODO: avoid hard coding
    num_pos = pos_classes

    # att_type: type_to_type
    att_type = np.concatenate(
        (
            np.linspace(0.99999, 0.999900, T_tilde),
            np.linspace(0.99990, 0.000009, time_step - T_tilde),
        )
    )
    att_type = np.concatenate(([1], att_type))
    at_type = att_type[1:] / att_type[:-1]
    att_type = np.concatenate((att_type[1:], [1]))

    # ctt_type: type_to_mask
    ctt_type = np.concatenate(
        (
            np.linspace(0.000009, 0.00009, T_tilde),
            np.linspace(0.000090, 0.99980, time_step - T_tilde),
        )
    )
    ctt_type = np.concatenate(([0], ctt_type))
    one_minus_ctt_type = 1 - ctt_type
    one_minus_ct_type = one_minus_ctt_type[1:] / one_minus_ctt_type[:-1]
    ct_type = 1 - one_minus_ct_type
    ctt_type = np.concatenate((ctt_type[1:], [0]))

    # btt_type: type_to_type
    btt_type = (1 - att_type - ctt_type) / type_classes
    bt_type = (1 - at_type - ct_type) / type_classes

    # att_pos: pos_to_pos
    att_pos = np.concatenate(
        (
            np.linspace(0.99999, 0.000100, T_tilde),
            np.linspace(0.00009, 0.000009, time_step - T_tilde),
        )
    )
    att_pos = np.concatenate(([1], att_pos))
    at_pos = att_pos[1:] / att_pos[:-1]
    att_pos = np.concatenate((att_pos[1:], [1]))

    # ctt_pos: pos_to_mask
    ctt_pos = np.concatenate(
        (
            np.linspace(0.000009, 0.00009, T_tilde),
            np.linspace(0.000100, 0.99990, time_step - T_tilde),
        )
    )
    ctt_pos = np.concatenate(([0], ctt_pos))
    one_minus_ctt_pos = 1 - ctt_pos
    one_minus_ct_pos = one_minus_ctt_pos[1:] / one_minus_ctt_pos[:-1]
    ct_pos = 1 - one_minus_ct_pos
    ctt_pos = np.concatenate((ctt_pos[1:], [0]))

    # btt_pos: pos_to_pos (instead of this btt, Gaussian matrix will be used when t < T_tilde)
    _btt_pos = 1 - att_pos - ctt_pos
    _btt_pos = np.concatenate(([0], _btt_pos))
    one_minus_btt_pos = 1 - _btt_pos
    one_minus_bt_pos = one_minus_btt_pos[1:] / one_minus_btt_pos[:-1]

    bt_pos = 1 - one_minus_bt_pos
    btt_pos = (1 - att_pos - ctt_pos) / num_pos

    bt_pos = np.concatenate(
        (bt_pos[:T_tilde], at_type[T_tilde:] / num_pos)
    )  ## <- Is this "at_type" correct?
    at_pos = np.concatenate(
        (at_pos[:T_tilde], (1 - ct_pos - bt_pos * num_pos)[T_tilde:])
    ).clip(min=1e-30)
    ct_pos = np.concatenate(((1 - at_pos - bt_pos)[:T_tilde], ct_pos[T_tilde:])).clip(
        min=1e-30
    )

    return dict(
        at_pos=at_pos,
        bt_pos=bt_pos,
        ct_pos=ct_pos,
        cumprod_at_pos=att_pos,
        cumprod_bt_pos=btt_pos,
        cumprod_ct_pos=ctt_pos,
        at_type=at_type,
        bt_type=bt_type,
        ct_type=ct_type,
        cumprod_at_type=att_type,
        cumprod_bt_type=btt_type,
        cumprod_ct_type=ctt_type,
    )
