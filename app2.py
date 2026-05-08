import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_bvp



st.title("Modeling dynamics: differential equations")
st.header("Mechanism Design and Price Discrimination")


st.markdown(r"""
### Economic Environment

Consumers differ in their **type** $\theta$, which measures how much they value a good.
The utility of a consumer buying the good with quality (or quantity) $x$ and paying price $p$ is

$$
u = \theta x - p
$$

The monopoly firm **cannot observe the consumer's type** but knows that type $\theta$ is distributed on interval
$[\underline{\theta}, \bar{\theta}]$ with density function $f(\theta)$.

To deal with this asymmetric information, the firm offers a **menu of contracts** for each $\theta$:

$$
(x(\theta), p(\theta))
$$

Each consumer $\theta$ chooses the option that maximizes utility:

$$
u(\theta) = \max_t \theta x(t) - p(t)
$$

In words, the consumer chooses option $(x(t),p(t))$ out of the menu to maximize $u$. In the language of mechanism design, type $\theta$ mimics type $t$.

What we want here is that consumer $\theta$ chooses the option that is meant for her: $t = \theta$. This is called "truthful revelation" of types.

Using the **envelope theorem**, truthful revelation implies:

$$
u'(\theta) = x(\theta)
$$

which links the consumers' utility or information rent to the quality/quantity they receive. The idea of "information rent" is that (all) consumers would have zero utility if the monopolist can observe their type and makes take-it-or-leave-it offers. In the latter case with perfect information, the monopolist would set $p(\theta) = \theta x(\theta)$ and leave no rents for consumers.


We can write the firm's profits on consumer type $\theta$ as:

$$
\pi(\theta) = p(\theta) - c(x(\theta)) = \theta x - u(\theta) - c(x(\theta))
$$

where $c(x)$ denotes the costs of producing quality (or quantity) $x$.


The optimization for the firm can be written as:

$$
\max \int_{\underline{\theta}}^{\bar \theta} f(\theta) (\theta x(\theta) - u(\theta) - c(x(\theta))) + \lambda(\theta) (u'(\theta)-x(\theta))d \theta
$$

On a "fraction" $f(\theta)$ of consumers, the firm makes $\pi(\theta)$ profits. We integrate $f(\theta) \pi(\theta)$ across all consumer types to find total (expected) profits. We add the truthful revelation constraint $u'(\theta) - x(\theta) = 0$ to the optimization problem with Lagrange multiplier $\lambda(\theta)$. For each type $\theta$ we add this constraint.

The first order (Euler) equations can be written as:

$$
(\theta - c'(x(\theta)))f(\theta) - \lambda(\theta) = 0
$$

and

$$
\lambda'(\theta) = -f(\theta)
$$

Further, the so called transversality condition implies $\lambda(\bar \theta) = 0$. Further, the firm has no interest in leaving rents to the lowest type: $u(\underline{\theta})=0$.

It is possible to solve this problem analytically, but here we solve it numerically using `scipy`.

Assume $c(x) = 0.5*cx^2$. Then the first order condition for $x$ can be written as $x = (\theta - \lambda(\theta)/f(\theta))/c$. This leads to two differential equations:

$$
u'(\theta) = \frac{1}{c}  (\theta - \lambda(\theta)/f(\theta))
$$

and

$$
\lambda'(\theta) = -f(\theta)
$$

with boundary conditions $u(\underline{\theta})=0$ and $\lambda(\bar \theta) = 0$.

This is called a boundary value problem `bvp` and we solve it numerically with the `scipy` function `solve_bvp`.


""")

st.markdown("### Model Parameters")

c = st.slider("Cost parameter (c)",0.1,3.0,1.0)
a=1
b=2

st.markdown("### Numerical Solution")


def f(t):
    return 1/(b-a)*np.ones_like(t)


def design(t,y):
    # t: Type
    # y[0]: u(t)
    # y[1]: lambda(t)
    return np.vstack([(t-y[1]/f(t))/c,-f(t)])


def bc(ya,yb):
    # ya[0]: u(\underline \theta) = 0
    # yb[1]: \lambda(\bar \theta) = 0
    return np.array([ya[0],yb[1]])

# type span
t_eval=np.linspace(a,b,100)

sol=solve_bvp(design,bc,t_eval,np.zeros((2,t_eval.shape[0])))
x=sol.yp[0]
p=t_eval*x-sol.y[0]

col1,col2=st.columns(2)

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(t_eval,x,label="Quantity x(θ)")
ax.plot(t_eval,p,label="Price p(θ)")
ax.plot(t_eval,x*t_eval,'--',label="Value θx(θ)")
# ax.scatter(t_eval[-1],x[-1],color="red")
ax.set_xlabel("Type θ")
ax.set_ylabel("Contract")
ax.set_title("Optimal Contract")
ax.legend()
ax.grid()

u=sol.y[0]

fig2,ax2=plt.subplots(figsize=(6,4))
ax2.plot(t_eval,u,label="Information rent u(θ)")
# ax2.scatter(t_eval[-1],u[-1],color="red")
ax2.set_xlabel("Type θ")
ax2.set_ylabel("Utility")
ax2.set_title("Information Rents")
ax2.legend()
ax2.grid()

with col1:
    st.pyplot(fig)

with col2:
    st.pyplot(fig2)

st.markdown(r"""
### Interpretation

Several important features appear in the optimal contract:

- **Higher types receive higher quality/quantity.** Consumers who value the product more
  are offered better quality or larger volume.

- **Prices increase with type.** High‑valuation consumers pay more because they get "more" (higher quality or higher quantity)

- Higher types obtain **positive information rents** because the firm must leave
  them enough surplus so they reveal their type truthfully.

### Exercises

- What is the relation between the left and right panel in the figure above? How can we see $u(\theta)$ in the left panel? And (this is a bit more subtle) how can we see $x(\theta)$ in the right panel of the figure?

- If we interpret $x$ as quantity, does the solution feature quantity discounts (price per unit descreases with more units bought)? Why (not)?

- Show that the welfare maximizing quality/quanity is given by $x^*(\theta) = \theta/c$. Is this the output schedule $x(\theta)$ that the monopolist offers in the solution above? Why (not)?
""")
