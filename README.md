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

Standard time-series analysis assumes parameters are largely static through time. I argue this is an invalid assumption, and will propose a partial (though numerically unstable) solution. At the end of the day, ARIMA/GARCH models are simple, interpretable and have a wide range of utilities. The aim is to address some of their inherent flaws and set up a more principled basis for forecasting.

### What Does Standard Time Series Analysis Look Like

Consider the foundational dynamic time series model, an $AR(p)$ process. The formulation of both the problem and its solution depends on the assumption that the process is sampled from the distribution $P(X_{t+1} = x | X_t, X_{t-1}, ..., X_{t-p},\phi)$, where $\phi \in \mathbb{R}^p$ and corresponds to the weighting terms on each components' contribution to $X_{t+1}$. Importantly, note that $\phi$ has no dependence on $t$. We will ignore initial conditions for now (where $t < p$) which is important for an unbiased solution, but not conceptually to the problem.

The textbook answer for how to solve this problem is the Yule-Walker equations. I propose a simpler intuition. Looking at Yule-Walker, you have the following:
$$
\vec{\gamma} = \mathbf{X} \vec{\phi} 
$$
where $\mathbf{X}$ is a stacked, shifted-by-one matrix of time series observations. A reasonable starting point to solve for $\phi$ would be least squares. If this looks an awful lot like solving $y=\mathbf{X}\beta$ for $\beta$ in linear regression, it's because it pretty much is. It turns out the "regression" in "autoregression" was the hint all along! Similar tricks apply to building intuitions around Koopman operators, but that's another topic for another day.

As simple as this seems, I've known of a lot of people who struggle making the jump to time series analysis despite knowing linear regression well enough. Making this identification yields a useful viewpoint because by stripping away the specialised language of time series analysis, we're free to think "it's just linear regression!" and understand more clearly the processes we're working with. (With the caveat of those initial conditions, which we're ignoring.) After all, everyone knows how to reason with linear regression. From here on, we will talk as if here $y = X_{t+1}$, $\hat{y} = \mathbb{E}[X_{t+1}]$ and $x_{[i]} = X_t,...,X_{t-p}, X \in \mathbb{R}^p$. From this perspective:
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

ARIMA models work pretty well, and they're simpler than we thought! So why would we choose to reinvent the wheel?

### Something Went Wrong

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

Firstly, decreasing the standard error of the mean estimate tells us nothing about the uncertainty over this estimate. Uncertainty estimates would give us some bounds on how different our params are from any given time period. Using a Bayesian approach gets us part of the way toward a solution - assuming away scalability and runtime. Accounting for uncertainty would give us more variance in our forecasts, but would be a step towards reliably-calibrated forecasts even if the bounds are large. Reliably calibrated forecasts are essential for forward risk pricing. However, the uncertainty parameter is also independent of time. As such, an approach whereby uncertainty is unconditionally estimated does not necessarily give us information about trends or correlations within the uncertainty. If we don't attempt to capture this information, we weaken our forecasts dramatically, without necessarily addressing the core problem. More on this in a moment.

Secondly, increasing the duration of our lookback window may stabilise the numerics, but makes the issue of distance between the "center" (where the params are most suitable) and the present even further. If we think of our dataset as a kind of "window" function, we can use additional time steps to stablilize the short-term deviations out of our estimate. This comes at the cost of increasing the "delay" of our system, as is often seen on the moving averages on trading charts.

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
Putting aside intuitions about window widths for a moment, LTI system theory has a rich literature on the kinds of delays we can expect. It also hints at solutions people have already employed to address problems.

So in summary, a static set of parameters yields an estimate (ceteris paribus) best suited for the middle of our time window. Model estimates degrade with time, and this is in part due to the tendency of most of our models to optimise to this center.Backtesting or train/test splits can sometimes reveal the presence of such a phenomenon, but can't necessesarily address it. Adding additional variance without considering where the variance belongs will only make forecasts less useable, even if they're more correct.

### A Solution? Or At Least, the Hints of One

Assume we have some data, spanning a few years. We begin by estimating the parameters for all the data at once, then estimating the parameters for the data one month at a time. 

NOTE: Add Images

The monthly estimates are noisy, but you can see that they tend to live around the overall mean, and tend to behave more similarly to nearby values than distant ones. **This is exactly what we'd expect to see if our parameters were varying through time, strongly conditional on previous values.** If there's sequence information (and we have reason to believe there is), and if we can capture it, then we can model the change in these parameters from time $t-T$ to time $t$ (the current moment). Our future forecasts are then made from parameters estimated from a point much closer in time, with attendant improvements in accuracy.

Because our parameters are noisy, it would be nice if we can have some accounting for long-tailed noise to help stabilise our estimates. Teams have proposed solutions, from the distribution of $y$ being set to something long-tailed (Student T, GEV or even Cauchy) to more robust estimators based on the L1 distance. Ideally we want our estimation process to correctly ascribe variance to the correct component, as simply adding in long-tailed distributions may cause more problems than they solve when simulating our forecasts forwards.

As a proposition and statement of intent;
> The parameters of an ARIMA-family model have a latent evolution through time, that has strong conditionality on previous values. These parameters should change slowly through time, except in dramatic breaks in behaviour. A model that captures these relationships will be able to produce better forecasts of future time series behaviours than a model fit to a single window of time.

I believe this process to be stochastic and unobservable, but can be estimated via the observable changes in ARIMA-like behaviour of the data. This isn't to say the parameter evolution is causal...it's still just a model. However, estimating these parameters should yield a better, more stable subspace to build probabilistic forecasts on.

### Major Components

So we have a few bits and pieces that we can bring together to solve this, as well as some attendant complications.
- We can use Bayesian inference to get estimates of uncertainty. This alone doesn't solve all of our problems, as exact methods (such as HMC) play poorly with large datasets. We can use variational inference, but we have to choose our approximating distribution carefully
- We can use repeated window functions to approximate the time evolution of these parameters, but this means choosing a window size. Picking a window size means we can't effectively respond to dynamic changes in expected timeframes, or learn to detect them.

Firstly, the windowing problem. I wanted to avoid having to pick a timescale, even if I would have to compute as if I had picked a timescale. The reason for this is mostly a question of maximising data through multiscale modelling  - I wanted to be able to let the behaviours at short and long timescales inform the expected behaviour, and allow for flexibility in training and inference. Currently we are still calculating the number of steps in the problem, as opposed to using IIR-like recursive methods. This can present an opportunity for improvement.

Addressing the variational inference problem, the reimplementation of GARCH in a Bayesian format was not too difficult. Getting a stable implementaton, however, was more difficult than expected. Many of the priors used to handle estimating parameters under the large variances of natural time series data cause major issues when simulating forwards. Trial and error revealed an implementation that only infrequently produced NaNs.

At some point I discovered the literature on neural stochastic differential equations. These were a backprop-enabled algorithm with Pytorch support, that (like Neural O/PDEs) allowed for dynamic timescale calculations. This got me thinking - is it possible that we model the evolution of the parameters directly as the ODE outputs? This is the implementation you'll find in the example above. As a theoretical side benefit, this approach offers greater scalability (as an amortised method) as well as trivial integration of exogenous variables.

### Major Issues
The above outlined approach has signs of working, but there are some significant flaws in the execution that I've yet to resolve
- Dynamically simulating variables through autoregression works poorly with deep learning, and tends to result in serious instability. I've yet to tap the drift diffusion literature and I'm hesitant to use some of the older hacks from the RNN literature (0-1 scaling etc.) as it undermines the push for a model as "clean" as vanilla ARIMA
- Variational inference does uncertainty estimates poorly. HMC may be necessary to get good solutions, but HMC doesn't scale well. Even approximate HMC methods appear to fail, despite significant decreases in sample quality.
- The various params definitely live on some manifold that isn't $\mathbb{R}^d$. Thinking about an AR parameter, increasing the 3rd coefficient for a fixed output $y$ doesn't make sense without some expected reduction in either the 2nd or 4th coefficient. I've yet to find a way to make this happen that isn't "let the neural net figure it out". I've encountered some hints in the literature that a "reparameterization trick" exists for autoregressive coefficients (i.e. the params all live in unbounded linear space, and some mapping pulls them onto the "AR Param Manifold"), but I've yet to see concrete implementations or even equations.
- Bounding the SDE. Even reformulating the SDE as an Ornstein-Uhlenbeck process to "pull" it towards 0 requires careful tuning and doesn't guarantee results. Worse still, the idea of learning the starting conditions of the SDE is either not viable, or converges to a solution extremely slowly
- Even accounting for all the above, even without exploding terms, a large part of the variance gets bundled into the evolution of the mean terms. Instead of a random model, you end up with something that appears more deterministic than stochastic. Perhaps a neural SDE would be better for estimating the time evolution of the parameters, or some other continuous time model. Relatedly, these values often collapse to 0

Ultimately, I think I need to go back to some of the tricks used to solve other similar SDEs, or remove degrees of freedom from the experiment. 