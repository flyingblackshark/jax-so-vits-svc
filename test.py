import jax
import jax.random
rng = jax.random.PRNGKey(1234)
print(rng)
m = jax.random.normal(rng,[8,8,8])
print(m) 