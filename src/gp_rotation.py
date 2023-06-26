import numpy as np
import pymc3 as pm
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
from celerite2.theano import terms, GaussianProcess
import arviz as az

__all__ = ["rotation_model",
           "sample_gp"]


def rotation_model(rtime, rflux, rflux_err, prot_init):

    x = np.ascontiguousarray(rtime, dtype=np.float64)
    y = np.ascontiguousarray(rflux, dtype=np.float64)
    yerr = np.ascontiguousarray(rflux_err, dtype=np.float64)
    mu = np.mean(y)
    y = (y / mu - 1) * 1e3
    yerr = yerr * 1e3 / mu

    # define pymc3 model 
    with pm.Model() as model:
        # The mean flux of the time series
        mean = pm.Normal("mean", mu=0.0, sigma=10.0)

        # A jitter term describing excess white noise
        log_jitter = pm.Normal("log_jitter", mu=np.log(np.mean(yerr)), sigma=2.0)

        # A term to describe the non-periodic variability
        sigma = pm.InverseGamma(
            "sigma", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        rho = pm.InverseGamma(
            "rho", **pmx.estimate_inverse_gamma_parameters(0.5, 2.0)
        )

        # The parameters of the RotationTerm kernel
        sigma_rot = pm.InverseGamma(
            "sigma_rot", **pmx.estimate_inverse_gamma_parameters(1.0, 5.0)
        )
        log_period = pm.Normal("log_period", mu=np.log(prot_init), sigma=2.0)
        period = pm.Deterministic("period", tt.exp(log_period))
        # period = pm.Uniform("period", lower=0.1, upper=10)
        log_Q0 = pm.HalfNormal("log_Q0", sigma=2.0)
        log_dQ = pm.Normal("log_dQ", mu=0.0, sigma=2.0)
        f = pm.Uniform("f", lower=0.1, upper=1.0)

        # Set up the Gaussian Process model
        kernel = terms.SHOTerm(sigma=sigma, rho=rho, Q=1 / 3.0)
        kernel += terms.RotationTerm(
            sigma=sigma_rot,
            period=period,
            Q0=tt.exp(log_Q0),
            dQ=tt.exp(log_dQ),
            f=f,
        )
        gp = GaussianProcess(
            kernel,
            t=x,
            diag=yerr**2 + tt.exp(2 * log_jitter),
            mean=mean,
            quiet=True,
        )

        # Compute the Gaussian Process likelihood and add it into the
        # the PyMC3 model as a "potential"
        gp.marginal("gp", observed=y)

        # Compute the mean model prediction for plotting purposes
        pm.Deterministic("pred", gp.predict(y))

        # Optimize to find the maximum a posteriori parameters
        map_soln = pmx.optimize()

    return model, map_soln


def sample_gp(model, map_soln, ncores=20, nchains=20):

    with model:
        trace = pmx.sample(
            tune=1000,
            draws=1000,
            start=map_soln,
            cores=ncores,
            chains=nchains,
            target_accept=0.9,
            return_inferencedata=True,
            random_seed=list(np.arange(ncores)),
        )

    return trace