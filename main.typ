#import "template.typ": *

#show: project.with(
  title: "Решение СЛАУ при помощи QIO",
  author: "Яременко Андрей",
)

= Description

Solving linear equation systems using quantum inspired optimization (QIO) for solving quadratic unconstrained binary optimization (QUBO)

= Formulas

$ A x = b <==> ||A x - b||^2 --> min <==> x^T A^T A x - 2 b^T A x --> min $
$ x =  A^T (A A^T)^(-1) b spa - spa "is one solution" $ 

$ A^T A "has n real non-negative eigenvalues" $
$ ||x||^2 = b^T (A A^T)^(-1) A A^T (A A^T)^(-1) b = b^T (A A^T)^(-1) b $
$ ||x||^2 <= ||b||^2 ||(A A^T)^(-1)|| <= (||b||^2)/("smallest singular value of " A A^T) spa - spa "bound on the solution" $
Generally there is no upper bound on the solution because A can be arbitrarily close to singular. And also we don't want to compute SVD or the inverse matrix.

= Algorithms

==  Initial algorithm

1. Find bounds on $x_i$
2. Folmulate initial problem in terms of quadratic optimization
3. Split the hypercube of possible solutions into $2^n$ parts and find their middle points ($x_i = x'_i + Delta_i q_i, spa q_i in {0, 1}$)
4. Solve QUBO problem (substitute new variables and use $0^2 = 0, spa 1^2 = 1$)
5. Update bounds and repeat.

Cons: Doesn't necessarily converge to an exact solution. Also has issues with errors in quantum computations.

Potential improvements: 

1. Choose points randomly
2. Run multiple times, subtract previous solution and scale up system to increase precision and reliability.

== Generalized algorithm from the paper

1. Find bounds on $x_i$
2. Folmulate initial problem in terms of quadratic optimization
3. Represent variables with finite precision ($x_i = (-2^p + 2^r) q_p + sum_(i=r)^(p-1) 2^i q_i$)
4. Solve QUBO problem with new variables

Cons: Number of variables grows quadratically with precision.