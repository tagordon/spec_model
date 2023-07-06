import jax
import jax.numpy as jnp

from flowMC.nfmodel.rqSpline import RQSpline
from flowMC.sampler.MALA import MALA
from flowMC.sampler.Sampler import Sampler
from flowMC.utils.PRNG_keys import initialize_rng_keys
from flowMC.nfmodel.utils import *

import tinygp
import inspect
import numpy as np

@tinygp.helpers.dataclass
class Multiband(tinygp.kernels.quasisep.Wrapper):
    amplitudes: jnp.ndarray

    def coord_to_sortable(self, X):
        return X[0]

    def observation_model(self, X):
        return self.amplitudes[X[1]] * self.kernel.observation_model(X[0])
    
class Model():
    
    def __init__(self, t, mean, terms, nbands, hold_params=[]):
        
        self.t = t
        self.mean = mean
        self.terms = terms
        self.nbands = nbands
        self.hold_params = hold_params
        self.trained_sampler = None
        self.production_sampler = None
        
        self.log_prob, self.labels = self._build_log_prob(
            self.t, 
            self.mean, 
            self.terms, 
            self.nbands, 
            hold_params=self.hold_params
        )
                
    def train(
        self, 
        data,
        position, 
        ball=0.001, 
        n_chains=20, 
        n_loops=100, 
        n_epochs=10, 
        learning_rate=1e-3, 
        batch_size=1000000, 
        step_size=5e-3,
        RQSpline_layers=4,
        RQSpline_hidden=[64, 64],
        seed=42
    ):
        
        n_dim = len(position)
        rng_key_set = initialize_rng_keys(n_chains, seed=seed)
        initial_position = position * (1 + jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * ball)
        
        sampler = self._build_sampler(
            data,
            n_dim,
            rng_key_set,
            n_chains=n_chains,
            n_loop_training=n_loops,
            n_loop_production=0,
            n_local_steps=10,
            n_global_steps=100,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            step_size=step_size,
            RQSpline_layers=RQSpline_layers,
            RQSpline_hidden=RQSpline_hidden,
        )
        
        sampler.sample(initial_position, data)
        self.trained_sampler = sampler
        
    def run_production(
        self, 
        data,
        position, 
        ball=0.001, 
        n_chains=20, 
        n_loops=100, 
        n_epochs=10, 
        learning_rate=1e-3, 
        batch_size=1000000, 
        step_size=5e-3,
        RQSpline_layers=4,
        RQSpline_hidden=[64, 64],
        seed=42,
        init_sampler=None
    ):
        
        if init_sampler is None:
            if self.trained_sampler is None:
                raise Exception('normalizing flow model should first be trained by running Model.train(data, initial_parameters)')
            else:
                init_sampler = self.trained_sampler
            
        params = init_sampler.state.params
        variables = init_sampler.variables
            
    
        n_dim = len(position)
        rng_key_set = initialize_rng_keys(n_chains, seed=seed)
        initial_position = position * (1 + jax.random.normal(rng_key_set[0], shape=(n_chains, n_dim)) * ball)
        
        sampler = self._build_sampler(
            data,
            n_dim,
            rng_key_set,
            n_chains=n_chains,
            n_loop_training=0,
            n_loop_production=n_loops,
            n_local_steps=50,
            n_global_steps=50,
            n_epochs=n_epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            step_size=step_size,
            RQSpline_layers=RQSpline_layers,
            RQSpline_hidden=RQSpline_hidden,
            model_init = {"params": params, "variables": variables}
        )
        
        sampler.sample(initial_position, data)
        self.production_sampler = sampler


    def _build_sampler(
        self, 
        data, 
        n_dim,
        rng_key_set,
        n_chains=20, 
        n_loop_training=0, 
        n_loop_production=0,
        n_local_steps=50,
        n_global_steps=50,
        n_epochs=10, 
        learning_rate=1e-3, 
        batch_size=100000, 
        step_size=5e3, 
        RQSpline_layers=4, 
        RQSpline_hidden=[64, 64], 
        model_init=None
    ):
        
        model = RQSpline(n_dim, RQSpline_layers, RQSpline_hidden, 8)
        local_sampler = MALA(self.log_prob, True, {"step_size": step_size}, use_autotune=True)
        
        sampler = Sampler(
            n_dim,
            rng_key_set,
            data,
            local_sampler,
            model,
            n_local_steps = n_local_steps,
            n_global_steps = n_global_steps,
            n_loop_training = n_loop_training,
            n_loop_production = n_loop_production,
            n_epochs = n_epochs,
            learning_rate = learning_rate,
            batch_size = batch_size,
            n_chains = n_chains,
            model_init = model_init
        )
        
        return sampler

    def _build_log_prob(self, t, mean, terms, nbands, hold_params=[]):
    
        # build array of mean parameters
        mean_params = inspect.getargspec(mean).args[1:]
        n_mean_params = len(mean_params)
    
        mean_inds = np.hstack(
            [i if mp in hold_params else [i] * nbands for i, mp in enumerate(mean_params)]
        )
        hold = list(np.hstack(
            [1 if mp in hold_params else 0 for mp in mean_params]
        ) )
        mean_params = list(np.hstack(
            [
                ["mean:" + mp] 
                if mp in hold_params 
                else ["mean:" + mp + "{0}".format(i) for i in range(nbands)] 
                for mp in mean_params
            ]
        ))
    
        # build array of gp parameters
        gp_params = []
        n_gp_params = []
        for i, term in enumerate(terms):
            params = list(inspect.signature(term).parameters)
            n_gp_params.append(len(params))
            [gp_params.append("kernel:" + p + "{0}".format(i)) for p in params]
        
        # array of random noise terms
        noise_params = ["diag{0}".format(i) for i in range(nbands)]
    
        # array of scale factors
        scales = ["scale{0}".format(i) for i in range(nbands - 1)]
    
        params = mean_params + gp_params + noise_params + scales
    
        @jax.jit
        def log_posterior(p, data):
        
            # unpack parameters
            idx = 0
            post_mean_params = p[idx:len(mean_params)]
            idx += len(mean_params)
            post_gp_params = p[idx:idx + len(gp_params)]
            idx += len(gp_params)
            post_noise_params = p[idx:idx + len(noise_params)]
            idx += len(noise_params)
            post_scales = p[idx:idx + len(scales)]
        
            # build kernel 
            idx = 0
            kernel = terms[0](*post_gp_params[idx:idx + n_gp_params[0]])
            idx += n_gp_params[0]
            for i, term in enumerate(terms[1:]):
                kernel += term(*post_gp_params[idx:idx + n_gp_params[i]])
                idx += n_gp_params[i]
        
            band_id = jnp.reshape(
                jnp.vstack(
                    [[jnp.ones(len(t), dtype=jnp.int32) * i] for i in range(nbands)]
                ).T
                , nbands * len(t)
            )
            x = jnp.sort(jnp.hstack([t] * nbands))
            X = (x, band_id)
        
            multiband_kernel = Multiband(
                kernel=kernel, 
                amplitudes=jnp.concatenate([jnp.array([1.0]), post_scales])
            )
        
            # unpack mean parameters
            mean_param_array = jnp.vstack(
                [jnp.tile(post_mean_params[np.where(mean_inds == i)], nbands) 
                 if h == 1 
                 else post_mean_params[np.where(mean_inds == i)] 
                 for i, h in enumerate(hold)
                ]
            )
            mean_arr = jnp.hstack(
                jnp.array([mean(t, *p) for p in mean_param_array.T]).T.flatten()
            )
                
            # build gp
            diags = jnp.tile(post_noise_params, len(t))
            gp = tinygp.GaussianProcess(multiband_kernel, X, diag=diags)
            Y = jnp.hstack(data.T)
            return gp.log_probability(Y - mean_arr)
    
        return log_posterior, params