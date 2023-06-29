#import torch
import jax
import flax
import jax.numpy as jnp

def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            # rl = rl.float().detach()
            # gl = gl.float()
            loss += jnp.mean(jnp.abs(rl - gl))

    return loss * 2

def l1_loss(real,fake):
        return jnp.sum(jnp.abs(real-fake))/len(real)
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        # dr = dr.float()
        # dg = dg.float()
        r_loss = jnp.mean((1 - dr) ** 2)
        g_loss = jnp.mean(dg**2)
        loss += r_loss + g_loss
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = jnp.mean((1 - dg) ** 2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses



def kl_loss(z_p, logs_q, m_p, logs_p, total_logdet, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_mask = z_mask.astype(jnp.int32)
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (jnp.exp(2.*logs_q)+jnp.square(z_p - m_p)) * jnp.exp(-2. * logs_p)
    kl = jnp.sum(kl * z_mask)
    # add total_logdet (Negative LL)
    kl -= jnp.sum(total_logdet)
    l = kl / jnp.sum(z_mask)
    return l

def kl_loss_back(z_p, logs_q, m_p, logs_p, z_mask):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    # z_p = z_p.float()
    # logs_q = logs_q.float()
    # m_p = m_p.float()
    # logs_p = logs_p.float()
    # z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * jnp.exp(-2.0 * logs_p)
    kl = jnp.sum(kl * z_mask)
    l = kl / jnp.sum(z_mask)
    return l
