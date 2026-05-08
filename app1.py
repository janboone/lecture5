import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

st.title("Modeling dynamics: differential equations")

st.header("Capital Accumulation and Economic Growth")

st.markdown(r"""
### The Solow Model

The Solow model is one of the fundamental models in macroeconomics. It explains how
**capital accumulation, population growth, and savings determine long‑run income per worker**.

The key dynamic equation describes how capital per worker evolves over time:

$$
\dot{k}(t) = s f(k(t)) - (n + \delta)k(t)
$$

where

- $k$ : capital per worker
- $\dot{k}$ : derivative of $k$ with respect to time
- $s$ : (exogenous) savings rate
- $f(k)$ : production function, usually $f(k) = k^{\alpha}$
- $n$ : population growth
- $\delta$ : depreciation rate

The term $s f(k)$ represents **investment**, while $(n+\delta)k$ represents the amount of
capital that must be devoted to **maintaining the capital stock**.

This is a differential equation with starting point $k_0$: capital level at $t=0$. This is called an initial value problem `ivp`.

The steady state occurs when

$$
\dot{k}=0
$$

meaning investment exactly offsets depreciation and population growth.

Below we solve the differential equation numerically using `scipy.integrate.solve_ivp` and
plot the transition of the economy over time.
""")

st.markdown("### Choose Parameters")

alpha = st.slider("Capital share (α)",0.1,0.9,0.3)
s = st.slider("Savings rate (s)",0.01,0.6,0.2)
n = st.slider("Population growth (n)",0.0,0.05,0.01)
delta = st.slider("Depreciation rate (δ)",0.0,0.1,0.05)
k0 = st.slider("Initial capital per worker (k₀)",0.01,5.0,0.5)

st.markdown("### Capital Dynamics")


def solow(t,k):
    return s*k**alpha - (n+delta)*k


k_ss=(s/(n+delta))**(1/(1-alpha))
t_span=(0,50)
t_eval=np.linspace(*t_span,200)
sol=solve_ivp(solow,t_span,[k0],t_eval=t_eval)


col1,col2=st.columns(2)

fig,ax=plt.subplots(figsize=(6,4))
ax.plot(sol.t,sol.y[0],label="Capital path k(t)")
ax.axhline(k_ss,linestyle="--",label="Steady state")
# ax.scatter([sol.t[-1]],[sol.y[0][-1]],color="red",zorder=3)
ax.set_xlabel("Time")
ax.set_ylabel("Capital per worker")
ax.set_title("Transition Dynamics")
ax.legend()

k_grid=np.linspace(0.01,max(5,k_ss+1),200)
invest=s*k_grid**alpha
break_even=(n+delta)*k_grid

fig2,ax2=plt.subplots(figsize=(6,4))
ax2.plot(k_grid,invest,label="s f(k)")
ax2.plot(k_grid,break_even,label="(n+δ)k")
ax2.scatter([k_ss],[s*k_ss**alpha],color="red",zorder=3,label="steady state")
ax2.set_xlabel("Capital k")
ax2.set_ylabel("Investment")
ax2.set_title("Solow Diagram")
ax2.legend()
ax2.grid()

with col1:
    st.pyplot(fig)

with col2:
    st.pyplot(fig2)

st.markdown(r"""
### Interpretation

- If $k_0$ is **below the steady state** capital level, investment exceeds depreciation
  and capital grows.

- If $k_0$ is **above the steady state** captal level, depreciation dominates and
  capital falls.

### Exercises

- Move the **savings rate slider** upward and observe how the steady state moves in
both graphs. Similarly, do this for $\alpha$ and $\delta$. Interpret the results.

- Solve analytically for the steady state capital level where $\dot{k}=0$.
""")
