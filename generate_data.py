import numpy as np
import jax.numpy as jnp
import jax
import tinygp
from spec_model import Multiband

def generate(t, mean_func, terms, nbands, offsets, amps, diags, mean_params, gp_params, seed=42):
    
    kernel = terms[0](*gp_params[0])
    for term, gpp in zip(terms[1:], gp_params[1:]):
        kernel += term(*gpp)

    band_id = jnp.reshape(jnp.vstack([[jnp.ones(len(t), dtype=jnp.int32) * i] for i in range(nbands)]).T, nbands * len(t))
    x = jnp.sort(jnp.hstack([t] * nbands))
    X = (x, band_id)
    multiband_kernel = Multiband(kernel=kernel, amplitudes=jnp.concatenate([jnp.array([1.0]), amps]))

    diag = jnp.tile(diags, len(t))
    mean = jnp.hstack(jnp.array([mean_func(t, *mp) + off for mp, off in zip(mean_params, offsets)]).T.flatten())
    gp = tinygp.GaussianProcess(multiband_kernel, X, diag=diag)

    return gp.sample(jax.random.PRNGKey(seed)) + mean