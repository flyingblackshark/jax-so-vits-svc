import jax.numpy as jnp

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.astype(jnp.float32)
    logs_q = logs_q.astype(jnp.float32)
    m_p = m_p.astype(jnp.float32)
    logs_p = logs_p.astype(jnp.float32)
    z_mask = z_mask.astype(jnp.float32)
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * jnp.exp(-2.0 * logs_p)
    kl = jnp.sum(kl * z_mask)
    l = kl / jnp.sum(z_mask)
    return l
