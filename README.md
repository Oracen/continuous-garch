# Continuous GARCH

Quick lib used to experiment with a garch model that has time-evolving parameters. The solutions tend to be degenerate, so I need to think about how these issues can be solved before it's in a publicly useable state.

## Basic Usage

```python
import pyro
import pandas as pd
import pyro.infer as infer

from continous_garch.model import GarchModel, GarchGuide


pyro.clear_param_store()
model_params = (1, 14, 7, 7, 3)
model = GarchModel(*model_params) # P, time-varying ARIMA with priors on params
guide = GarchGuide(*model_params) # Q approximating, using an SDE to model evolution

df = pd.read_csv("../data/SPY.csv")[["Close"]].apply(np.log).diff().fillna(0)
data = torch.Tensor(df.Close.values).reshape(-1, 1)

optimizer = optim.ClippedAdam({"lr": 1e-2, "betas": (0.99, 0.999)})
svi = infer.SVI(
    model,
    guide,
    optimizer,
    loss=infer.TraceGraph_ELBO(
        num_particles=16,
        vectorize_particles=True,
    ),
)
for i in range(2000):
    elbo = svi.step(data)
    if i % 5 == 0:
        print("Epoch: {}  Elbo loss: {}".format(i, elbo / check_data.shape[0]))
```

## Motivation

Standard time-series analysis assumes parameters are largely static through time. I argue this is an invalid assumption, and will propose a partial (though numerically unstable) solution.

### What Does Standard Time Series Analysis Look Like

Consider the foundational dynamic time series model, an $AR(p)$ process. The formulation of both the problem and its solution depends on the assumption that the process is sampled from the distribution $ P(X_{t+1} = x | X_t, X_{t-1}, ..., X_{t-p},\phi)$, where $\phi \in \mathbb{R}^p$ and corresponds to the weighting terms on each components' contribution to $X_{t+1}$. Importantly, note that $\phi$ has no dependence on $t$. We will ignore initial conditions for now (where $t < p$) which is important for an unbiased solution, but not conceptually to the problem.

The textbook answer for how to solve this problem is the Yule-Walker equations. I propose a simpler intuition. Looking at Yule-Walker, you have the following:
$$
\vec{\gamma} = \mathbf{X} \vec{\phi} 
$$
where X is a stacked, shifted-by-one matrix of time series observations.A reasonable starting point to solve for $\phi$ would be least squares. If this looks an awful lot like solving for $\beta$ in linear regression, it's because it pretty much is. It turns out the "regression" in "autoregression" was the hint all along! Similar tricks apply to understanding Koopman operators, but that's another topic for another day.

As simple as this seems, I've known of a lot of people who struggle making the jump to time series analysis despite knowing linear regression well enough. From here on, we will talk as if here $y = X_{t+1}$, $\hat{y} = \mathbb{E}[X_{t+1}]$ and $x = X_t,...,X_{t-p}, x \in \mathbb{R}^p$. This is a useful viewpoint because by stripping away the specialised language of time series analysis, we're free to think "it's just linear regression!" and understand more clearly the processes we're working with. (With the caveat of those initial conditions, which we're ignoring.) Everyone knows how to reason with linear regression. Taking this perspective:
- adding a bias is equivalent to drift terms
- log prices/returns correspond to multiplicative models, though perhaps this is less surprising
- regessing on the residuals of the predictins gives you an MA(q) process
- regressing on the finite/fractional differences of observations corresponds to an ARIMA(p,d,q) proces, with d being integer/real depending on the differencing method
- a mixture model with random effects on special dates/time periods looks like SARIMA(p,d,q,P,D,Q)
- concatenating the observation matrix with additional predictors and predicting multiple outputs is a standard vector autoregression with exogenous predictors;

and so on.

Evidently we don't need to consider hundreds of unique names, but rather pay attention to how our matrix $X$ and $y$ is constructed. We can therefore drop the parameters (p, q, d etc.) as they are modelling choices rather than qualitative differences. We can similarly generalise this way of thinking to $ARCH$ models. Assume $y = X_{t+1}$, and $y \sim Normal(\mu,\sigma)$. Taking this point of view;
- ARIMA models are a regression on the expectation of the mean $\mathbb{E}[\mu]$, 
- ARCH models are a strictly-positive regression on the expectated variance $\sigma$, often denoted $\mathbb{E}[\sqrt{\sigma^2}]$
- GARCH models combine estimates of both $\mu$ and $\sigma$

### What Went Wrong

We have "established" that these time series walk and talk a lot like linear regresssions. (Duck-typed homomorphisms?) But modulo dataset construction, time series models also inherit a lot of the assumptions regarding distributional assumptions Recall we said earlier: 

> Importantly, note that $\phi$ has no dependence on $t$.

Considering what this means from the point of view of our data, where our dependence on $t$ means $X$ lacks exchangeable indices. With exchangeability we assume that new datapoints will come from the same joint distribution $P(X=x,Y=y)$. This means we can assume the parameters apply equally to all datapoints, and will always be approximately equally wrong. The "middle" of the parameters is the "middle" of the distribution in some least squares sense (subject to sampling error).

```
fitted mu = 5

datapoint   | L1 error from mu
------------------------------
    a=5     |       0
    b=7     |       2
    c=2     |       3
    d=7     |       2
    e=4     |       1
    f=3     |       2
    g=7     |       2

can be reshuffled in a valid way to any permutation

datapoint   | L1 error from mu
------------------------------
    c=2     |       3
    f=3     |       2
    e=4     |       1
    a=5     |       0
    g=7     |       2
    b=7     |       2
    d=7     |       2
------------------------------
Predictions:
------------------------------
    h=4     |       1
    i=8     |       3

No clear trending behaviour in our letter indices
Our new observations look a lot like our past observations
```

Under time series conditions, _this is not remotely true_. The joint is always indexed by t, and in most cases $X_{t+1}$ will look more like $X_t$ than $X_{t-N}$ for even small $N$. Our parameters are still in the middle of our data distribution, but our sampling process looks very different. In practice, this means we're regressing on the "middle" time values, with past and future values being less well represented. Given we typically want to forecast future values rather than run historical counterfactuals, we would want that our parameters best reflect the present point in time $t$.

```
fitted mu = 5

datapoint   | L1 error from mu
------------------------------
    x_1=2   |       3
    x_2=3   |       2
    x_3=4   |       1
    x_4=5   |       0
    x_5=6   |       1
    x_6=7   |       2
    x_7=8   |       3
------------------------------
Predictions:
------------------------------
    x_8=9   |       4
    x_9=10  |       5

SEQUENCE MATTERS, forecast error constantly increasing
(as observed in most time series modelling)
```

The cases where this problem show up fairly commonly when reading the literature, but I suspect industry may be different. When demonstrating forecasting models (especially seasonal models with an annual component), I often see academics or bloggers using some very long time period to maximise the data used in the estimate of their AR params. Naturally, this has the effect of reducing the variance of the MLE estimate standard error.

This approach is flawed for 2 main reasons.

Firstly, decreasing the standard error of the mean estimate tells us nothing about the uncertainty over this estimate. Uncertainty estimates would give us some bounds on how different our params are from any given time period. Using a Bayesian approach gets us part of the way toward a solution - assuming away scalability and runtime - but fails to give us information about trends in the uncertainty. More on this in a moment.

Secondly, increasing the duration of our lookback window may stabilise the numerics, but makes the issue of distance between the "center" (where the params are most suitable) and the present even further. If we think of our dataset as a kind of "window" function, we're essentially smoothing the short-term deviations out of our estimate. Putting aside intuitions about time for a moment, LTI system theory has a rich literature on the kinds of delays we can expect, as well as exact ways to calculate them. 

```
fitted mu = 7 

datapoint   | L1 error from mu
------------------------------
    x_1=2   |       5
    x_2=3   |       4
    x_3=4   |       3
    x_4=5   |       2
    x_5=6   |       1
    x_6=7   |       0
    x_7=8   |       1
    x_8=9   |       2
    x_9=10  |       3
    x_10=11 |       4
    x_11=12 |       5
------------------------------
Predictions:
------------------------------
    x_8=9   |       6
    x_9=10  |       7

Distance from fitted param is actually getting worse
```

So in summary, a static set of parameters yields an estimate (ceteris paribus) best suited for the middle of our time window. Model estimates degrade with time, and this is in part due to the tendency of most of our models to optimise to this center.Backtesting or train/test splits can sometimes reveal the presence of such a phenomenon, but can't necessesarily address it.

### A Solution? Or At Least, the Hints of One

This can, and should, be understood within the context of LTI systems theory. One sensible 