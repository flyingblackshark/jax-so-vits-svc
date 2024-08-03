#import torch
import jax
import flax
import jax.numpy as jnp





# def kl_loss(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
#     """
#     z_p, logs_q: [b, h, t_t]
#     m_p, logs_p: [b, h, t_t]
#     total_logdet: [b] - total_logdet summed over each batch
#     """
#     kl = logs_p - logs_q - 0.5
#     kl += 0.5 * ((z_p - m_p) ** 2) * jnp.exp(-2.0 * logs_p)
#     kl = jnp.where(z_mask,kl,0)
#     kl = jnp.sum(kl)
#     # add total_logdet (Negative LL)
#     kl -= jnp.sum(total_logdet)
#     l = kl / jnp.sum(z_mask)
#     return l

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * jnp.exp(-2.0 * logs_p)
    kl = jnp.sum(kl * z_mask)
    l = kl / jnp.sum(z_mask)
    return l
