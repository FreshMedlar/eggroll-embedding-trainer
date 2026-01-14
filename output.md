### **Evolution Strategies at the Hyperscale**



**Bidipta Sarkar** _[∗]_ [1] _[,]_ [2] **, Mattie Fellows** _[∗]_ [1] **, Juan Agustin Duque** _[∗]_ [2] _[,]_ [3] **,**
**Alistair Letcher** _[†]_ [1] **, Antonio León Villares** _[†]_ [1] **, Anya Sims** _[†]_ [1] **, Dylan Cope** _[†]_ [1] **, Jarek Liesen** _[†]_ [1] **,**
**Lukas Seier** _[†]_ [1] **,Theo Wolf** _[†]_ [1] **, Uljad Berdica** _[†]_ [1] **, Alexander David Goldie** [1] _[,]_ [2] **,**
**Aaron Courville** [3] _[,]_ [5] **, Karin Sevegnani** [4] **, Shimon Whiteson** _[‡]_ [2] **, Jakob Nicolaus Foerster** _[‡]_ [1]

1 FLAIR - University of Oxford, 2 WhiRL - University of Oxford, 3 MILA– Québec AI Institute
4 NVIDIA AI Technology Center, 5 CIFAR AI Chair
{bidipta.sarkar,matthew.fellows,jakob.foerster}@eng.ox.ac.uk
juan.duque@mila.quebec, shimon.whiteson@cs.ox.ac.uk


**Abstract**



We introduce Evolution Guided General Optimization via Low-rank Learning (EGGROLL),
an evolution strategies (ES) algorithm designed to scale backprop-free optimization to large
population sizes for modern large neural network architectures with billions of parameters.
ES is a set of powerful blackbox optimisation methods that can handle non-differentiable or
noisy objectives with excellent scaling potential through parallelisation. Naïve ES becomes
prohibitively expensive at scale due to the computational and memory costs associated
with generating matrix perturbations _E ∈_ R _[m][×][n]_ and the batched matrix multiplications
needed to compute per-member forward passes. EGGROLL overcomes these bottlenecks
by generating random matrices _A ∈_ R _[m][×][r]_ _, B ∈_ R _[n][×][r]_ with _r ≪_ min( _m, n_ ) to form a
low-rank matrix perturbation _AB_ _[⊤]_ that are used in place of the full-rank perturbation _E_ .
As the overall update is an average across a population of _N_ workers, this still results in
a high-rank update but with significant memory and computation savings, reducing the
auxiliary storage from _mn_ to _r_ ( _m_ + _n_ ) per layer and the cost of a forward pass from
_O_ ( _mn_ ) to _O_ ( _r_ ( _m_ + _n_ )) when compared to full-rank ES. EGGROLL’s efficiency results in a
hundredfold increase in training throughput for billion-parameter models at large population
sizes, nearly reaching the throughput of pure batch inference. A theoretical analysis reveals
our low-rank update converges to the full-rank update at a fast _O_ - 1 _r_ - rate. Our experiments

show that (1) EGGROLL does not compromise the performance of ES in tabula-rasa RL
settings, despite being faster, (2) it is competitive with GRPO as a technique for improving
LLM reasoning, and (3) EGGROLL enables stable pre-training of nonlinear recurrent
language models that operate purely in integer datatypes. Code is available at our website:
[https://eshyperscale.github.io/](https://eshyperscale.github.io/)


Rank-one perturbation _Ei_ Fitness evaluation Weighted average



Initial weights




      

= _f_


      

= _f_



+ _σ_


+ _σ_











Final rank- _N_ update


## ... ...


      


= _f_



+ _σ_



(a)


Figure 1: Schematic visualization of EGGROLL using _N_ workers.


*Equal Contribution _†_ Core Contributor, sorted by alphabetical order in first names _‡_ Equal Senior Authors


1


8


7


6


5


4



100


80


60


40


20



Normalized Training Speeds



Pure Integer Pretraining: Test Loss


0 200 400 600 800 1000
Training Step


(b)



10 [5]


10 [4]


10 [3]


10 [2]


10 [1]



EGGROLL PPO OpenES


(a)



Figure 2: (a) Relative speed of our method, EGGROLL, in terms of experience throughput versus prior methods, where
100 represents the maximum batch throughput of pure inference. See Appendix F for more details. (b) We use EGGROLL
to train RNN language models from scratch using only integer datatypes, scaling population size from 64 to 262144.


**1** **Introduction**


Evolution Strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) are an attractive
alternative to first-order methods based on gradient backpropagation, for several reasons. First, ES does not
require differentiability, so it can optimise a broader class of models, like models with discrete parametrisation
spaces (cellular automata), and can optimize objectives where gradients are unavailable or noisy, like outcomeonly rewards in LLM fine-tuning (Qiu et al., 2025). Second, ES is more robust to noisy and ill-conditioned
optimisation landscapes (Wierstra et al., 2011; Xue et al., 2021). Unlike gradients, population-based exploration
smooths irregularities (Salimans et al., 2017), tolerates discontinuities, and mitigates issues like ill-conditioned
curvature or vanishing and exploding gradients in long-range or recurrent settings (Hansen, 2023). Third,
ES is highly amenable to scaling through parallelisation, since fitness evaluations are independent across
population members and require only the communication of scalar fitnesses, which maps cleanly onto modern
inference infrastructure and yields near-linear speedups on large clusters (Salimans et al., 2017). By contrast,
backpropagation requires communicating and aggregating gradients across devices, yielding updates with high
memory and computation costs. Additionally, backpropagation requires special care when training models
with low-precision datatypes, whereas ES can directly optimize any model with the same datatypes used
at inference time. Together, these properties position ES as a potentially powerful foundation for training
large, discrete, or hybrid architectures, and end-to-end systems with non-differentiable components, including
large language models (LLMs) (Brown et al., 2020; Chowdhery et al., 2023; Du et al., 2022; Fedus et al.,
2022).


Despite this potential, there are practical obstacles to employing ES at scale. In deep learning architectures
(Goodfellow et al., 2016), the majority of trainable parameters form linear mappings represented by matrices
(Rosenblatt, 1962; Hochreiter & Schmidhuber, 1996; Bengio et al., 2000; Krizhevsky et al., 2012; Goodfellow
et al., 2014; Kingma & Welling, 2014; Vaswani et al., 2017); naïvely adapting ES therefore requires generating
full-rank matrix perturbations that replicate the entire parameter set for every population member. This inflates
memory costs and forces frequent movement of large weight tensors. Evaluating these perturbations then
requires a separate sequence of matrix multiplications per member, so the total compute and wall-clock
time scale roughly with the population size and sequence length. In billion-parameter regimes, these two
costs dominate, making it difficult to scale ES beyond small models or small populations (Qiu et al., 2025;
Korotyshova et al., 2025).


To mitigate both memory and computational bottlenecks, we introduce Evolution Guided General Optimization
via Low-rank Learning (EGGROLL), an ES algorithm that allows for the efficient training of neural network
architectures with billions of parameters. Analogous to LoRA’s low-rank adapters in gradient-based training
(Hu et al., 2022), EGGROLL generates _low-rank_ parameter-space perturbations for ES: instead of sampling
a full-rank matrix _E ∈_ R _[m][×][n]_, we sample _A ∈_ R _[m][×][r]_ and _B ∈_ R _[n][×][r]_ with _r ≪_ min( _m, n_ ) and form
_E_ = ~~_√_~~ 1 _AB_ _[⊤]_ . This reduces auxiliary perturbation matrix storage from _mn_ to ( _m_ + _n_ ) _r_ per layer, and
_r_
proportionally reduces tensor movement. Moreover, we use a counter-based deterministic random number
generator (RNG) (Salmon et al., 2011; Bradbury et al., 2018) to reconstruct noise on demand, so matrix


2


perturbations need not persist in memory. When evaluating the fitness of members of multiple perturbations
in parallel, EGGROLL batches a population of low-rank adapters and shares the base activations, enabling
a single forward pass that applies all _AB_ _[⊤]_ updates via specialized batched matrix multiplications. The
compute required beyond standard batched inference scales with _O_ ( _r_ ( _n_ + _m_ ) _N_ ), where _N_ is the population
size, instead of _O_ ( _nmN_ ) for standard ES, yielding substantial memory and inference savings and making
EGGROLL efficient for extremely large models. We emphasise that our method does not restrict the updates
to be low-rank: the overall EGGROLL update is an average of rank _r_ matrices across the population, making
the matrix parameter update rank min( _Nr, m, n_ ) .


EGGROLL is described in detail in Section 4. We provide a rigorous theoretical analysis of the low-rank
approximation accuracy in Section 5, proving that EGGROLL updates converge to the full rank Gaussian
ES updates at an _O_ ( [1] _/r_ ) rate. This fast convergence rate suggests that low-rank updates are sufficient to
train large-scale architectures, even with _r ≪_ min( _m, n_ ). In our extensive empirical evaluation, we test this
hypothesis across a wide range of domains in Section 6. In tabula rasa and multi-agent RL (MARL) settings,
we show that EGGROLL does not compromise performance compared to naïve ES despite being faster.
To demonstrate the scalability of our method for LLM fine-tuning, we conduct experiments on pretrained
RWKV7 (Peng et al., 2025) models, modern recurrent language models that enable large batch inference
due to their constant state size. Finally, we develop a nonlinear RNN language model that operates purely in
integer datatypes, and demonstrate that EGGROLL can stably pretrain this language model, a feat which is
only feasible due to the large population sizes enabled by EGGROLL.


**2** **Preliminaries**


_All proofs for theoretical results in this paper are found in the Appendix._


**2.1** **Low Rank Matrix Approximations**


When adapting high dimensional foundation models for specific tasks, updating the parameters using gradient
based methods has high memory requirements. LoRA uses low-rank approximations to the matrix multiplications to reduce these costs. For each matrix _Mi ∈_ R _[m][×][n]_ in the model, a low-rank approximation can be made
by decomposing each matrix:

_Mi ≈_ _Mi_ [0] [+] _[ A][i][B]_ _i_ _[⊤][,]_

where _Mi_ [0] [:=][ StopGrad][(] _[M][i]_ [)][ is the imported matrix from the foundation model with frozen parameters and]
_Ai ∈_ R _[m][×][r]_ and _Bi ∈_ R _[n][×][r]_ are low-width column matrices (i.e., _r ≪_ min( _m, n_ )) whose parameters are
updated through gradient-based optimisation during task-specific adaptation. This reduces the number of
optimisation parameters for each matrix from _mn_ to _r_ ( _m_ + _n_ ). In this paper, we use a similar low-rank matrix
approximation for evolutionary strategies.


**2.2** **Gaussian Matrix Distribution and Matrix Norms**


In this paper, we focus on evolution strategies that target _matrix parameters_ . Many variables we study are
Gaussian distributed. When working in matrix space, it is convenient to use the matrix Gaussian distribution
(Dawid, 1981), which is defined directly over matrices _X ∈_ R _[m][×][n]_ :



1
_N_ ( _M, U, V_ ) = ~~_mn_~~
(2 _π_ ) 2 det( _U_




  
_−_ [1]
2 [exp] 2



~~_n_~~ ~~_m_~~
2 det( _U_ ) 2 det( _V_ ) 2




  -  - [�]

[1] _V_ _[−]_ [1] ( _X −_ _M_ ) _[⊤]_ _U_ _[−]_ [1] ( _X −_ _M_ ) _,_

2 [tr]



where _M ∈_ R _[m][×][n]_ is the mean matrix, _U ∈_ R _[m][×][m]_ is the row covariance matrix and _V ∈_ R _[n][×][n]_ is the
column covariance matrix. The matrix Gaussian distribution is a generalisation of the multivariate Gaussian
distribution _N_ ( _µ,_ Σ) defined over vector space. Sampling a matrix _X ∼N_ ( _M, U, V_ ) from a Gaussian matrix
distribution is equivalent to sampling a vector vec( _X_ ) _∼N_ ( _µ,_ Σ) from a multivariate Gaussian distribution
with mean _µ_ = vec( _M_ ) and covariance matrix Σ = _V ⊗_ _U_ where _⊗_ denotes the Kronecker product. For
isotropic matrix Gaussian distributions with covariance matrices _U_ = _σ_ [2] _Im_ and _V_ = _σ_ [2] _In_, the equivalent
multivariate Gaussian distribution is also isotropic with Σ = _σ_ [2] _Imn_ . To measure distance between matrices,
we use the Frobenius norm:

~~��~~
_∥M_ _∥F_ := _mi,j_ [2] _,_


_i,j_


3


which provides an upper bound on the matrix 2-norm (Petersen & Pedersen, 2008). We denote the _ℓ_ [2] vector
norm as _∥·∥_ .


**2.3** **Evolution strategies**


Evolution strategies (ES) (Rechenberg, 1978; Beyer, 1995; Beyer & Schwefel, 2002) is a set of blackbox
methods for optimising general systems. ES has emerged as a useful alternative to gradient-based methods,
particularly when a system is noisy or non-differentiable. Our problem setting focuses on fitness functions
whose parameters are matrices: our goal is to find a matrix _M_ _[⋆]_ _∈_ R _[m][×][n]_ that maximises the fitness function
_M_ _[⋆]_ _∈_ arg max _M_ _∈_ R _m×n f_ ( _M_ ). In comparison to gradient-based methods which use derivatives of the
function _f_ ( _M_ ) to update the parameters _M_ directly, evolutionary methods update a population distribution
over the parameter space. This is achieved by learning a set of parameters _θ_ to maximise the expected fitness
_f_ ( _z_ ) under a population distribution _π_ ( _M_ _|θ_ ):


_J_ ( _θ_ ) = E _M_ _∼π_ ( _M_ _|θ_ ) [ _f_ ( _M_ )] _._ (1)


Taking derivatives of _J_ ( _θ_ ) yields the gradient:


_∇θJ_ ( _θ_ ) = E _M_ _∼π_ ( _M_ _|θ_ ) [ _∇M_ log _π_ ( _M_ _|θ_ ) _f_ ( _M_ )] _,_


which is used to update the population distribution’s parameters using (stochastic) gradient ascent with a
suitable stepsize _αt_ :


_θt_ +1 _←_ _θt_ + _αt∇θJ_ ( _θt_ ) _._ (2)


_∇M_ log _π_ ( _M_ _|θ_ ) is known as the score function, which avoids taking gradients directly through the fitness
function in Eq. (1).


**2.4** **Gaussian Matrix ES**

In this paper, we study ES using Gaussian policies: _π_ ( _M_ _|θ_ ) = _N_ ( _µ, Imσ_ [2] _, Inσ_ [2] ). In addition to its
mathematical convenience, the central limit theorem means that the Gaussian distribution emerges naturally
from a low-rank approximation as rank increases, even if the matrices _A_ and _B_ are themselves non-Gaussian.
We assume that _σ_ [2] is fixed and the ES algorithm optimises over the mean matrix _θ_ = _µ_ which acts as a proxy
for the true maximum of the fitness function. By adapting well-known derivations (Wierstra et al., 2011) to the
Gaussian matrix case, we derive the ES gradient for our problem setting:
**Proposition 1.** _Using the Gaussian matrix policy π_ ( _M_ _|µ_ ) = _N_ ( _µ, Imσ_ [2] _, Inσ_ [2] ) _the ES objective can be_
_written as:_


_J_ ( _µ_ ) = E _E∼P_ ( _E_ ) [ _f_ ( _M_ = _µ_ + _σE_ )] _._ (3)


_and the matrix gradient of the ES objective is:_

_∇µJ_ ( _µ_ ) = _−_ _σ_ [1] [E] _[E][∼][P]_ [ (] _[E]_ [)][ [] _[E][ ·][ f]_ [(] _[M]_ [ =] _[ µ]_ [ +] _[ σE]_ [)]] _[ .]_ (4)


_where P_ ( _E_ ) _is a zero-mean standard normal p_ ( _E_ ) = _N_ (0 _, Im, In_ ) _. Moreover, the natural matrix gradient is_
_equal to σ_ [2] _∇µJ_ ( _µ_ ) _, i.e. equivalent to Eq._ (4) _up to a factor of σ_ [2] _._


We see from Eq. (4) that Gaussian matrix ES methods optimise the objective _J_ ( _µ_ ) by generating search
matrices _E ∼_ _P_ ( _E_ ) from a standard matrix normal distribution _N_ (0 _, Im, In_ ) around the parameter matrix
_θ_ = _µ_ .


We remark that for Gaussian population distributions, the ES update in Eq. (4) is equal to the natural evolution
stategy (NES) update up to a factor of _σ_ [2] . NES (Wierstra et al., 2008; 2011) updates follow the _natural gradient_
(Amari, 1998; Kakade, 2001) of the objective in Eq. (1). This means that for our problem setting where _σ_ is
assumed fixed and is absorbed into the stepsize in Eq. (2), Gaussian matrix ES and NES are equivalent. A key
benefit of the natural gradient is that it takes into account the local geometry of the underlying parameter space
when updating the search distribution, making the updates invariant to the choice of parametrisation.


4


**3** **Related Work**


**3.1** **Evolutionary Algorithms**


Evolutionary algorithms have long been a compelling alternative to backpropagation-based training methods.
Despite encompassing a broad range of algorithms (e.g., genetic algorithms (Such et al., 2018) or symbolic
evolution (Koza, 1994)), most contemporary research on evolution has shifted towards algorithms that scale
better to large numbers of neural network parameters (Jaderberg et al., 2017; Hansen & Ostermeier, 2001;
Salimans et al., 2017).


Our work focuses on evolving the weights of predetermined architectures, building up from the NES (Wierstra
et al., 2011) family of methods. Such an approach has recently grown in prominence, following the application
of NES to policy learning in conventional RL environments (Salimans et al., 2017) to alleviate some of the
challenges that policy-gradient methods struggle with, like long-horizon environments. Since then, evolution
has been widely applied in other domains, such as meta-learning (e.g., (Lu et al., 2022; Metz et al., 2022; Lange
et al., 2023; Goldie et al., 2024; 2025)), hyperparameter tuning (e.g., (Parker-Holder et al., 2021; Tani et al.,
2021; Vincent & Jidesh, 2023)) and drug discovery (Towers et al., 2025). Here, we consider the limitations
and solutions to applying ES at significant scale, beyond the small networks and population sizes of these prior
works, with a broad focus on policy-learning. In particular, Salimans et al. (2017) uses a maximum population
size of 1440, whereas our maximum population size is on the order of hundreds of thousands.


ES is limited by its requirement for full, potentially expensive evaluations of the fitness function due to
simulating policies in long-horizon environments and potentially high memory usage. Persistent Evolution
Strategies (Vicol et al., 2021) demonstrate a significant speedup by updating networks _online_ (i.e., during the
unroll), with followups offering further variance reduction (Li et al., 2023b; Vicol et al., 2023). We note that
these works are orthogonal to our focus on scaling up the population size of ES; we leave the application of
these techniques on top of EGGROLL to future work.


**3.2** **Evolution Strategies for LLMs**


Although gradient backpropagation is typically used for LLM training and fine-tuning, prior work has explored
ES variants for fine-tuning. In particular, zeroth order optimization (Zhang et al., 2024), which is analogous to
ES with a population size of 1, is used by Malladi et al. (2023) for memory-efficient LLM fine-tuning. Yu et al.
(2025) extend this approach by projecting perturbations to a low-rank subspace, improving the convergence
of zeroth order optimization. Jin et al. (2024) performs ES directly on LoRA matrices. These works focus
on the supervised fine-tuning setting, finding comparable performance to full fine-tuning, but they do not
determine whether pretraining is possible with zeroth order methods; we find that large population sizes are
necessary for pretraining performance, indicating that zeroth order optimization methods would be unsuitable
for pretraining.


Recent work has also explored ES in the context of LLM reasoning. Korotyshova et al. (2025) first train
LoRA adapters using supervised fine-tuning (SFT) before decomposing them into fixed SVD bases alongside
singular values that are trained using CMA-ES. They achieve comparable performance to GRPO (Shao et al.,
2024) in significantly less wall-clock time on math reasoning benchmarks. Qiu et al. (2025) directly use ES
to optimize all LLM parameters for reasoning, with stronger performance than GRPO on the countdown
reasoning task. However, both of these approaches use relatively small population sizes, on the order of a
hundred unique perturbations per update, and instead collect hundreds of rollouts per perturbation to efficiently
use GPUs. By contrast, our approach allows all generations to use different perturbations, such that our
maximum population size per update is orders of magnitude larger (equal to the maximum inference batch
size), without compromising token generation throughput.


**4** **EGGROLL**


We now introduce and motivate our method EGGROLL, which is presented in Algorithm 1. In Section 4.1
we derive a low-rank ES update that approximates a full-rank ES gradient. One practical issue with using
a low-rank matrix approximation is that its distribution and score function have no analytic solution except
for degenerate cases, so in Section 4.2 we derive an alternative score function from the limiting high-rank
Gaussian which we propose as an approximation.


5


**4.1** **Low Rank Evolution Strategies** **Algorithm 1** EGGROLL( _r, α, σ, T_ max _, N_ workers)



Recall the Gaussian matrix ES update from Eq. (4).
Our goal is to introduce a tractable approximation to
generating full-rank matrices, using low-rank matrices
_AB_ _[⊤]_ as our search matrices instead. We denote the
distribution of _A_ as _p_ ( _A_ ) and _B_ as _p_ ( _B_ ) assume that the
elements of _A_ and _B_ are drawn independently:
**Assumption 1.** _Assume all elements ai,j ∈_ _A and_
_bi,j ∈_ _B are continuous identically and independently_
_distributed random variables according to some zero-_
_mean, symmetric, absolutely continuous (i.e., it has a_
_density) distribution p_ 0( _·_ ) _with finite 4th order moments_
_and variance_ 0 _< σ_ 0 [2] _[.]_



**initialise** _µ_ and workers with known random seeds _ς_
**for** _T_ max timesteps **do**

**for** Each worker _i ∈{_ 1 _, . . . N_ workers _}_ in parallel **do**

_Ai ∼_ _p_ ( _Ai_ ) _, Bi ∼_ _p_ ( _Bi_ )
_Ei ←_ ~~_√_~~ 1 _r_ _AiBi_ _[⊤]_
_fi ←_ _f_ ( _µ_ + _σEi_ )
**end for**
Workers share scalar fitness _fi_ with other workers
**for** Each worker _i ∈{_ 1 _, . . . N_ workers _}_ in parallel **do**

Reconstruct _Ej_ for _j ∈{_ 1 _, . . . N_ workers _}_ from _ς_

_µ ←_ _µ_ + _α_ _N_ Workers1  - _Nj_ =1Workers _Ej_ _fj_
**end for**
**end for**



The low-rank approximation _AB_ _[⊤]_ must be carefully integrated into the Gaussian matrix ES objective in
Eq. (3) to ensure that the variance of the updates remains bounded with increasing number of columns _r_ as the
elements of _AB_ _[⊤]_ are a sum of _r_ random variables. To counter this, we scale the outer product by ~~_√_~~ 1 which
_r_
keeps the variance of _E_ = ~~_√_~~ 1 _AB_ _[⊤]_ as _O_ (1).
_r_


Observe that _E_ = ~~_√_~~ 1 _AB_ _[⊤]_ maps to the manifold M _[r]_ _⊂_ R _[m][×][n]_ of rank _r_ matrices, which is a subset of R _[m][×][n]_ .
_r_
This means that the density _p_ ( _E_ ) is defined with respect to a unit volume over the manifold and cannot be
defined with respect to the standard unit volume in Euclidean space as the volume of the manifold is zero
using this measure. For the corresponding score function, gradients with respect to log _p_ ( _E_ ) are defined
over the tangent space to M _[r]_ instead of in the usual Euclidean space. An intuitive reason for this is that
there is dependence between elements of _E_ and so it is not possible to take partial derivatives with respect
to each element whilst keeping all other elements free. In principle it is possible to define all distributions
and gradients with respect to the manifold M _[r]_, however this makes an analysis of low-rank updates difficult.
Instead, we study the variable _Z_ = ~~_√_~~ 1 _AB_ _[⊤]_ + _ϵ_ where _ϵ_ is an _m × n_ matrix with independent decaying
_r_
Gaussian elements _ϵi,j ∼N_ (0 _,_ _[σ]_ _ϵ_ [2] _/r_ ). The variable _Z_ = ~~_√_~~ 1 _r_ _AB_ _[⊤]_ + _ϵ_ has a well-defined density _p_ ( _Z_ ) for all
_σϵ >_ 0. This conceptual tool greatly simplifies analysis and as _σϵ_ can be made arbitrarily small and hence _p_ ( _E_ )
can be arbitrarily close to _p_ ( _Z_ ), the difference between the update used in practice and the update analysed is
negligible. Substituting for _p_ ( _E_ ) = _p_ ( _Z_ ) in Eq. (3), we derive the low-rank ES objective:


_J_ LR( _µ_ ) = E _Z∼p_ ( _Z_ ) [ _f_ ( _M_ = _µ_ + _σZ_ )] _._


We now verify that _p_ ( _z_ ) is well-defined and show how _J_ LR( _µ_ ) can be optimised using the score function of
_p_ ( _Z_ ):
**Theorem 1.** _**Low-rank ES Gradient.**_ _Let Assumption 1 hold. Then for finite r:_

_gLR_ := _∇µJLR_ ( _µ_ ) = _−_ _σ_ [1] [E] _[Z][∼][p]_ [(] _[Z]_ [)][ [] _[∇][Z]_ [ log] _[ p]_ [ (] _[Z]_ [)] _[ f]_ [(] _[M]_ [ =] _[ µ]_ [ +] _[ σZ]_ [)]] _[ .]_


_where Z is generated by sampling A ∼_ _p_ ( _A_ ) _, B ∼_ _p_ ( _B_ ) _, ϵ ∼_ _p_ ( _ϵ_ ) _and making the transformation Z_ =
~~_√_~~ 1 _AB_ _[⊤]_ + _ϵ._
_r_


**4.2** **Score Function Approximation**

As the mapping ( _A, B, ϵ_ ) : _→_ _Z_ = ~~_√_~~ 1 _AB_ _[⊤]_ + _ϵ_ is non-invertible, there does not exist a simple closed form
_r_
solution for the density _p_ ( _Z_ ) and score function except in degenerate cases such as _m_ = _n_ = 1. Instead, we
use an approximation for the score function _S_ [ˆ] ( _Z_ ) _≈∇Z_ log _p_ ( _Z_ ):

_g_ ˆLR = _−_ _σ_ [1] [E] _[Z][∼][p]_ [(] _[Z]_ [)]              - _S_ ˆ( _Z_ ) _f_ ( _M_ = _µ_ + _σZ_ )� _._ (5)


To optimise the ES objective using the EGGROLL update, we adapt the parallelised evolutionary strategies
algorithm from Salimans et al. (2017). In our experiments and Algorithm 1, we use a Gaussian approximate


6


score function, which is obtained from taking the limit _r →∞_ . As _ϵ_ is a random matrix formed of independent
Gaussian elements, it can be decomposed into an equivalent sum of _r_ independent Gaussian matrices:



1
_ϵ_ = ~~_√_~~
~~_r_~~



_r_

- _ϵi,_


_i_ =1



where each random matrix _ϵi ∼_ _p_ ( _ϵi_ ) has the same distribution as _ϵ_ . Likewise, the matrix _AB_ _[⊤]_ can be
decomposed as a sum of independent, zero-mean vector outer products:



_AB_ _[⊤]_ =



_r_

- _aib_ _[⊤]_ _i_ _[,]_

_i_ =1



where _ai_ and _bi_ are the _i_ th column vectors of _A_ and _B_ . This allows us to write _Z_ as a standardised sum of _r_
independent random matrices:



1
_Z_ = ~~_√_~~
~~_r_~~



_r_

- - _aib_ _[⊤]_ _i_ [+] _[ ϵ][i]_ - _._ (6)

_i_ =1



Under Assumption 1, the central limit theorem proves that _p_ ( _Z_ ) converges in distribution to a Gaussian
_N_ (0 _, Imσ_ 0 [4] _[, I][n][σ]_ 0 [4][)][. Using this limiting distribution in place of the true distribution] _[ p]_ [(] _[Z]_ [)][, we obtain the]
Gaussian approximate score function:


_S_ ˆ( _Z_ ) = _−_ [1] _Z._ (7)

_σ_ 0 [4]


We remark that EGGROLL is not wedded to any particular score function approximator and we derive and
explore a set of mean-field approximators in Appendix B as alternatives, however our experiments show that
the Gaussian approximator has the best overall performance on the set of tasks we evaluated on. We make
a Monte Carlo estimate of expectation in Eq. (5) with _N_ workers samples to optimise the parameters _µ_ using
(approximate) stochastic gradient ascent. This yields the Gaussian EGGROLL update:









Here we have absorbed the constants _σ_ [1] [and] _σ_ 10 [4] [into the tunable learning rate] _[ α][t]_ [. As each random matrix]

_Ei,t_ in Eq. (8) has rank _r_ almost surely and the matrix is updated using a sum of _N_ worker such matrices,
the overall EGGROLL matrix parameter update has rank min( _Nr, m, n_ ) almost surely, meaning that the
overall parameter update is not restricted to be low-rank. For all experiments in Section 6, _Nr >_ min( _, m, n_ ),
meaning EGGROLL parameter updates are full-rank.


**4.3** **Hardware-Efficient EGGROLL Implementation**


A key reason to use EGGROLL over standard ES is that large populations can be simulated in parallel on a
GPU thanks to the low-rank perturbations. For the sake of exposition, we write equations from the perspective
of a single worker, _i_, and explain in text how this corresponds to batched GPU operations.


Consider the task of computing a batched forward pass over inputs _xi ∈_ R _[d][in]_ for a linear layer with mean
parameter _µ ∈_ R _[d][out][×][d][in]_ . The standard forward pass is just a regular matrix multiplication, _xµ_ _[T]_, since _µ_ is
constant across all threads. However, naively applying ES by trying to compute _xi_ ( _µ_ + _σEi_ ) _[T]_ becomes a
batched matrix multiplication, which is inefficient on GPUs since every element of _µ_ + _σEi_ is only used in a
single multiplication, yielding poor arithmetic intensity.


7


However, with EGGROLL we know that _xi_ ( _µ_ + _σEi_ ) = _xiµ_ + ~~_√_~~ _σr_ ( _xiBi_ ) _A_ _[T]_ _i_ [. In this context, the bulk of]
compute is spent on the efficient calculation of _xiµ_ using regular matrix multiplication. Meanwhile, when
_r_ = 1, _xiBi_ simply becomes an inexpensive batched vector-vector dot product to get a batch of scalars, which
is then processed by a batched scalar-vector multiplication when multiplying by _A_ _[T]_ _i_ [. This decomposition]
is key to efficient batched LoRA inference, such as those used by vLLM (Kwon et al., 2023), which why
EGGROLL achieves the same speeds as batched LoRA inference systems.


We additionally optimize the update process by not explicitly materializing the individual _Ei_ in the computation
of [�] _i_ _[N]_ =1 _[E][i][f][i]_ [, the key term in the Gaussian approximate score function. In particular, when the rank is 1, we]
reconstruct _A ∈_ R _[N]_ _[×][d][out]_ and _B ∈_ R _[N]_ _[×][d][in]_ and calculate the expression as ( _A ⊙_ _f_ ) _B_ _[T]_, which is a simple
matrix multiplication.


**5** **Approximation Analysis**


We now analyse how fast the Gaussian score approximation from Eq. (7) converges to the true Gaussian
ES matrix gradient in Eq. (4). We introduce the following formal regularity assumption for the fitness
function:
**Assumption 2.** _Assume that f_ ( _M_ ) _is bounded, that is_ sup _M |f_ ( _M_ ) _| < ∞._


Denote the true full-rank Gaussian ES gradient as _g_ True := _∇µJ_ ( _µ_ ). Our key theoretical result characterises
the error rate between the Gaussian score approximator in the low-rank update ˆ _g_ LR _[r]_ [from Eq. (][5][) and the true]
gradient using the matrix Frobenius norm:
**Theorem 2.** _Let Assumptions 1 and 2 hold and set σ_ 0 = 1 _, then:_




      - 1
_∥g_ ˆLR _[r]_ _[−]_ _[g]_ [True] _[∥][F]_ [=] _[ O]_




_._ (9)



_r_



The convergence rate in Eq. (9) is faster than the
typical _O_ ( [1] _/_ ~~_[√]_~~ _r_ ) rate dictated by the general parametric central limit theorem. Our analysis shows that
this is due to the symmetry in our problem under
Assumption 1. To obtain our results, we make an
Edgeworth expansion (Bhattacharya & Ranga Rao,
1976) of the density _p_ ( _Z_ _[r]_ ), which expands _p_ ( _Z_ _[r]_ )
as the limiting Gaussian distribution plus a sum of
decaying terms that are controlled by the 3rd order
and higher cumulants of _p_ ( _Z_ _[r]_ ). Each _i_ th order cumulant term is multiplied by a factor that decays at

   -   rate _O_ _r_ _[−]_ _[i][−]_ 2 [2] . For symmetric zero-mean distribu

|r=1|Col2|Col3|Col4|Col5|Col6|
|---|---|---|---|---|---|
|~~r=2~~<br>r=3<br>r=5<br>r=10||||||
|r=50<br>r=100<br>r||||||
|||||||
|||||||
|||||||



Figure 3: Plot of Marginal Score Multiplied by Density for
Increasing _r_










   -   rate _O_ _r_ _[−]_ _[i][−]_ 2 [2] . For symmetric zero-mean distribu
Figure 3: Plot of Marginal Score Multiplied by Density for

tions, all odd cumulants are zero (for the same reason

Increasing _r_

that all odd moments of a symmetric distribution are
zero). Hence, the rate of convergence to the limiting distribution is controlled by the 4th order term, which has
rate _O_ - _r_ _[−]_ [1][�] .



Although the full distribution _p_ ( _Z_ _[r]_ ) has no general closed-form solution, the distribution over marginals
_p_ ( _zi,j_ ) is more amenable to analysis. We derive _p_ ( _zi,j_ ) for generalised Gaussian distributed _ai,j_ and _bi,j_ in
Section B. To illustrate the fast converge rate, we plot the negative density _×_ score function _p_ ( _zi,j_ ) _zi,j_ for the
marginal distribution _p_ ( _zi,j_ ) in Fig. 3 using Gaussian distributed _ai,j_ and _bi,j_ with _σ_ 0 [2] [= 1][ (see Theorem][ 4][ for]

                                 - 2                                  
a derivation). The figure shows that _p_ ( _zi,j_ ) _zi,j_ quickly converges to the limiting function ~~_√_~~ _[z][i,j]_ 2 _π_ [exp] _−_ _[z][i,j]_ 2,

recovering the Gaussian form from the true natural policy gradient update. Even at _r_ = 1, the function is not a
poor approximation. After _r_ = 10, the function has nearly converged and after _r_ = 50, the function is visually
indistinguishable from the limit, providing evidence for the hypothesis that the low-rank approximation is
accurate even for very low rank regimes _r ≪_ min( _m, n_ ).




_[i,j]_ 2 _π_ [exp] - _−_ _[z][i,j]_ 2 2



a derivation). The figure shows that _p_ ( _zi,j_ ) _zi,j_ quickly converges to the limiting function ~~_√_~~ _[z][i,j]_



8


**6** **Experiments**


**6.1** **Pure Integer Pretraining of an RNN Language Model**


To demonstrate the potential of EGGROLL as a general optimization method, we study whether EGGROLL
could be used for language model pretraining. Since EGGROLL does not rely on gradients, we can explicitly
design a language model architecture to be efficient and hardware-friendly at inference time. In particular, we
build a model under the following constraints to emphasize the flexibility of EGGROLL:


1. **Pure Integer Training:** On H100 systems, int8 is the fastest datatype, with int8 matrix multiplication
with int32 accumulation being the fastest tensor core operation. Furthermore, integer datatypes are
much simpler to implement in hardware, providing massive energy savings for high-throughput
systems (Horowitz, 2014). Therefore, we keep all weights in int8 and all activations in integer
formats, _never_ casting to floating point at any point during training.


2. **Nonlinear RNN:** Modern language models use sequence-parallel architectures like Transformers and
SSMs, since they enable stable gradients without backpropagation through time. However, most of
these sequence-parallel architectures are unable to handle simple state tracking (Merrill et al., 2024),
whereas classic recurrent networks like LSTMs and GRUs can handle these problems with a single
layer. Since EGGROLL does not require backpropagation through time, we can train on unbounded
sequence lengths (Li et al., 2023a) with nonlinear RNNs of broader complexity classes. Specifically,
we develop a variant of the minGRU model (Heck & Salem, 2017) that performs all operations in
integer formats.


3. **Removal of all Activation Functions:** Inspired by Foerster (2017), we remove all activation
functions, like the rectified linear unit and hyperbolic tangent, due to the nonlinearity present in the
int8 datatype. Specifically, the saturated addition of int8 values provides sufficient nonlinearity due to
the implicit clipping of values to the int8 dynamic range, which evolution strategies can exploit.


We call the resulting language model EGG, the **E** volved **G** enerative **G** RU, an EGGROLL-friendly architecture.
Its architecture is similar to standard pre-layernorm transformer decoder models, but we (1) use a variant of L1
normalization instead of L2 normalization for our layernorm to avoid square roots, (2) replace self-attention
with our custom GRU, and (3) perform all operations in integer datatypes. See Appendix C for more details on
the architecture.


We train an EGG model with 6 layers and hidden dimension 256 to do character-level prediction on the
minipile dataset (Kaddour, 2023). We update parameters after 100 tokens for each population member,
applying truncated ES by keeping the hidden state and only resetting at document boundaries. We plot the
test loss in Fig. 2b over training steps across a range of population sizes, where the best test loss is 3.41
bits/byte. We find that training is stable and loss curves are relatively smooth, especially with large population
sizes, avoiding loss spikes, nan values, and other instabilities associated with backprop-based training at low
precision datatypes.


Note that our largest population size is 2 [18] = 262144, which is two orders of magnitude larger than the largest
experiment done by Salimans et al. (2017) while only requiring a single GPU to train. We see that multiplying
the population size by 8 results in the loss dropping by approximately 0.4 over the range of population values
we have tested, though this pattern will eventually break since the loss must be strictly positive. We conduct
more ablations in Appendix E, determining how data efficient training can be achieved with EGGROLL and
validating the importance of large batch sizes.


**6.2** **Reinforcement Learning Tasks**


In these experiments, we compare the performance of EGGROLL against standard OpenES as implemented in
Salimans et al. (2017) on reinforcement learning tasks. Given the small network sizes, we can use Open ES at
this scale, but we note that as network sizes increase, the use of vanilla OpenES becomes impossible. We use
the standard formulation of simply optimizing for the final return in the environment. For both EGGROLL
and OpenES, we perform hyperparameter optimization (HPO) separately for each environment. For each
algorithm–environment pair, we define plausible ranges for all key hyperparameters based on prior work and
preliminary experiments. We then perform 20 random search trials, where each trial corresponds to a single


9


Pendulum-v1



1.0


0.5


0.0


1.0


0.5


0.0



0 2 4
Steps 1e8


Jumanji 2048


0 2 4
Steps 1e8



Craftax Symbolic


0 2 4
Steps 1e8


Navix DoorKey (8x8)


0 2 4
Steps 1e8





Brax Inverted Double Pendulum


1.0


0.5


0.0

0 2 4
Steps 1e8


Kinetix Thrust Control Left (m)


1.5


1.0


0.5


0.0

0 2 4
Steps 1e8



1.0


0.5


0.0


1.0


0.5


0.0



Figure 4: Comparison of reinforcement learning Mean returns normalized by PPO performance for 10 seeds. The returns
are evaluated using the mean of the parameters. HPO was conducted for each algorithm/environment pair. The shaded
region is the standard error of the mean .


training run with a randomly sampled hyperparameter configuration. Each configuration is evaluated based on
the final return achieved by the mean policy parameters at the end of training. After all trials, we select the
configuration that yields the highest final return. Using this best configuration, we then run 10 independent
seeds to evaluate performance and report the mean and standard error of the mean across these seeds.


We use policy networks with 3 layers of 256 neurons and a range of environments that demonstrate different
capabilities. We evaluate across the Navix (Pignatelli et al., 2024), Craftax (Matthews et al., 2024), Brax
(Freeman et al., 2021), Kinetix (Matthews et al., 2025), and Jumanji (Bonnet et al., 2024) suites of environments.
We evaluate 16 environments in total. To pick environments, we choose environments that are not trivial or
impossible for PPO to solve, according to the original papers. We also choose environments that are part of
different categories when these are available (e.g. environment size in Kinetix or categories in Jumanji).


We show a subsample of the environments that were evaluated in Fig. 4. The remaining environment results
are in Appendix G.1. Our findings show that EGGROLL is competitive with Open ES on 7/16 environments,
underperforms on 2/16, and outperforms on 7/16. This does not take into account the speed-ups when
compared to using OpenES (full-rank updates). We postulate that the reason for this performance increase is
that the large networks are difficult to optimize for Open ES and lend themselves well to low rank updates. All
hyperparameter configuration details are available in Appendix G.1.


**6.3** **LLM Fine-tuning for Reasoning Tasks**


We apply EGGROLL for LLM finetuning of RWKV-7 (Peng et al., 2025) models in two reasoning tasks:
countdown and GSM8K. The RWKV architecture is a recurrent model that, compared to transformers, is
especially suited to parallelization due to the fact that any memory otherwise spent on the KV cache can be
used to evaluate population members. The training curves of EGGROLL and GRPO in countdown are shown
in figure 5a. EGGROLL fine-tuning on an RWKV-7 1.5B model converges to a higher validation accuracy
of 35% (v.s. 23%) under the same hardware and wall-clock time in the countdown task. Similarly, figure 5b


10


GSM8K — RWKV 7g7B


0 2 4 6 8 10
Relative wall-clock time (hours)


GRPO (n=3) EGGROLL (n=3)


(b)



0.3


0.2


0.1


0.0



Countdown — RWKV 7g1.5B


0 1 2 3 4 5 6 7 8
Relative wall-clock time (hours)


GRPO (n=3) EGGROLL (n=3)


(a)



0.80


0.75


0.70


0.65


0.60



Figure 5: Comparison of the validation score of 3 seeds of EGGROLL v.s. 3 seeds of GRPO in: (a) Countdown task with
an RWKV 7g1.5B model on a single GPU. EGGROLL allows 1024 parallel generations per GPU whereas GRPO only 32.
(b) GSM8K task with an RWKV 7g7B model on 8 GPUs. EGGROLL allows 8096 parallel genrations (1024 per GPU)
whereas GRPO only 256 (32 per GPU).


shows that EGGROLL outperforms GRPO on GSM8K fine-tuning. Our scoring function draws parallels to the
group relative advantage of GRPO. In particular, to score a set of noise directions, _E ≡{E_ 1 _, . . ., En}_, we first
compute their accuracies, _{s_ 1 _,qi, . . ., sn,qi}_, on _|q|_ = _m_ questions, creating a matrix of scores _S ∈_ R _[m][×][n]_ .
We then compute the _z_ score per question, with the main difference that we use the global variance ¯ _σ_, and
average over all the questions to compute the final score for the noise direction _Ei_ :



_s_ ¯ _i_ = [1]

_m_



_m_





- _zi,qj_ = [1]

_m_

_j_ =1



_m_



_m_



_j_ =1



_si,j_ _−_ _µqj_ _._

_σ_ ¯



This scoring function has the purpose of weighting all questions within the same batch the same across
population members.


**7** **Conclusion**


In this paper, we introduce EGGROLL, a powerful method for blackbox optimisation that scales evolutionary
strategies to billion-parameter models and beyond using low-rank search matrices. Our experiments demonstrate that EGGROLL is effective with rank as small as _r_ = 1, which represents substantial computational and
memory savings for negligible decrease in performance when compared to the full-rank ES update. Empirically,
EGGROLL delivers large speedups over naïve ES in tabula rasa and multi-agent RL, and can power end-to-end
training pipelines for large language models. Our theoretical analysis reveals that the low-rank EGGROLL
update quickly converges with rank _r_, however further theoretical analysis is needed to explain the success of
our method when _r_ = 1.


Looking forward, we are working on applying EGGROLL for other problems beyond the reach of modern
gradient-based techniques. In particular, EGGROLL can enable the training of large scale end-to-end neurosymbolic systems (Sarker et al., 2021) which have nondifferentiable components. For instance, we can
train neural networks that directly interface with symbolic modules for specialized functions, like memory
or calculations. We can also optimize end-to-end systems of language models, training them to be aware of
inference-time harnesses and interactions with other agents in complex systems.

