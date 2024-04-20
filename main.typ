#import "template.typ": *

#show: project.with(
  title: "SLE via QIO",
  author: "Andrew Yaremenko",
)

= Description

Solving linear equation systems using quantum inspired optimization (QIO) for solving quadratic unconstrained binary optimizatio problem (QUBO)

= Formulas

$ A x = b <==> ||A x - b||^2 --> min <==> x^T A^T A x - 2 b^T A x --> min $
$ x =  A^T (A A^T)^(-1) b $ 

$ ||x||^2 = b^T (A A^T)^(-1) A A^T (A A^T)^(-1) b = b^T (A A^T)^(-1) b $
$ ||x||^2 <= ||b||^2 ||(A A^T)^(-1)|| <= (||b||^2)/("smallest singular value of " A A^T) $

= Algorithm 1

+ Formulate SLE as quadratic minimization problem
+ Pick initial bounds for each coordinate
+ Substitute $x_i = l b_i + (u b_i - l b_i)/2^p (1/2 + q_(i 1) + 2 q_(i 2) + 4 q_(i 3) + ... + 2^(p-1) q_(p-1 i))$
+ Solve QUBO for $q_(i j)$
+ Update bounds with the neighbourhood of found solution
+ Repeat untill bounds are small enough

== Improvement 1
After finding a solution $x_0$, substitute $y = x + x_0$ and repeat the algorithm.

== Improvement 2

Add perturbation to step 3. Choose the constant and the coefficients of $q_(i j)$ with the following procedure:
```py
  rnd = ss.truncnorm(-1 / 2 / sigma, 1 / 2 / sigma, 1 / 2, sigma).rvs()
  mn, mx = rnd, rnd
  const = rnd * lengths[i] + lb[i]
  coefs = []
  for k in range(prec):
      rnd = ss.truncnorm(-mn / sigma, (1 - mx) / sigma, 0, sigma).rvs()
      coefs += [(rnd + 2 ** k) * lengths[i]]
      mn = min(mn, mn + rnd)
      mx = max(mx, mx + rnd)
```