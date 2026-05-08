import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp

st.title("Modeling dynamics: differential equations")

st.header("Pricing Dynamics with Learning by Doing")

st.markdown(r"""
### Economic Idea

In many industries firms become more efficient as they produce more. This is known as
**learning by doing**.

Cumulative production lowers costs because firms gain experience with the technology.

We model this by assuming that constant marginal costs depend on cumulative output $Q$:

$$
c(Q) = \frac{c}{1 + Q}
$$

In words, producing $q$ units today --with cumulative output $Q$ from previous periods-- costs $c q/(1+Q)$.

The relation between $Q_t$ and $q_t$ is given by $Q'_t = q_t$: the change in cumulative $Q$ at time $t$ is given by $q_t$.

Demand is given by

$$
p(q) = 1 - q
$$

Firms therefore decide **how much to produce today** knowing that producing more
now reduces costs in the future.

This creates a **dynamic optimization problem** over the period $[0,T]$:

$$
\max \int_0^T e^{-\rho t} (p(q_{t})-c(Q_t))q_{t} - \lambda_t (Q'_t - q_{t}) dt
$$
where $\rho$ denotes the discount factor and $\lambda_t$ the Lagrange multiplier on the constraint $Q'_t = q_t$.

For this problem the first order (Euler) equations can be written as:

$$
e^{-\rho t}(p(q_t)-c(Q_t) + p'(q_t) q_t) + \lambda_t = 0
$$

and

$$
-\lambda'_t = -e^{-\rho t} q_t C'(Q_t)
$$

Further we have the initial condition $Q_0 = 0$ (when the firm starts there is no cumulative output from previous periods) and end (transversality) condition $\lambda_T = 0$.


Verify that with $p = 1-q, c(Q) = c/(1+Q)$, we can solve the first order condition for $q$ as

$$
q_t = 0.5*(1-c(Q_t)+e^{\rho t} \lambda_t)
$$

Then our system of differential equations becomes:

$$
Q'_t = 0.5(1-c(Q_t) + e^{\rho t} \lambda_t)
$$

and

$$
-\lambda'_t = -e^{-\rho t} c'(Q_t) Q'_t = 0.5*e^{-\rho t} \frac{c}{(1+Q_t)^2} (1-c(Q_t) + e^{\rho t} \lambda_t)
$$

with $Q_0 = \lambda_T = 0$.

This is a boundary value problem that we solve numerically with `solve_bvp`.


""")

st.markdown("### Model Parameters")

rho = st.slider("Discount rate (ρ)",0.0,1.0,0.1)
# c = st.slider("Cost parameter (c)",0.1,1.0,0.5)
c = 0.1

st.markdown("### Solving the Dynamic System")


def learning(x,y):
    # x is time t
    # y = Q, lambda
    return np.vstack((
        0.5*(1-c/(1+y[0])+np.exp(rho*x)*y[1]),
        -np.exp(-rho*x)*c/(1+y[0])**2*0.5*(1-c/(1+y[0])+np.exp(rho*x)*y[1])
    ))


def bc(ya,yb):
    # ya[0] = Q_0; ya refers to initial condition; index [0] refers to Q: y[0]
    # yb[1] = lambda_T; yb refers to end condition; index [1] refers to lambda: y[1]
    return np.array([ya[0],yb[1]])


t_span=(0,10)
t_eval=np.linspace(*t_span,400)
sol=solve_bvp(learning,bc,t_eval,np.zeros((2,t_eval.shape[0])))

q=sol.yp[0]
Q=sol.y[0]
l = sol.y[1]
t=sol.x

col1,col2=st.columns(2)

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(t,q,label="Dynamic q(t)")
ax.plot(t,(1-c/(1+Q))/2,':',label="Static optimum")
ax.set_xlabel("Time")
ax.set_ylabel("Output")
ax.set_title("Optimal Output Path")
ax.legend()
ax.grid()

fig2,ax2=plt.subplots(figsize=(6,4))
ax2.plot(t,l,label="$\\lambda_t$")
# ax2.plot(t,Q,label="Optimal cumulative Q(t)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Shadow Price")
ax2.set_title("Value of Learning Over Time")
ax2.legend()
ax2.grid()

with col1:
    st.pyplot(fig)

with col2:
    st.pyplot(fig2)

st.markdown(r"""
### Interpretation

- We define the static optimum as the monopoly output level with given marginal costs $c/(1+Q)$ where $Q$ follows the optimal path derived above. Conditional on $Q$, the left panel of the figure shows the difference between a static profit maximizer and a firm taking the future value of learning into account.

- At the beginning of the time horizon the firm produces **more than the static monopoly output**. The reason is that producing more increases cumulative experience $Q$ and therefore reduces future costs.

- This extra production is an **investment in learning**. The right panel shows the marginal value of this learning.


### Exercises

- Increase the **discount rate**. What happens to output $q$ and why?

- Why do the dynamic and static solutions coincide at $T=10$?
""")
