from jax_md import simulate, rigid_body, quantity
from jax.tree_util import tree_map, tree_reduce
from jax_md.rigid_body import RigidBody

"""Simulation Single Dispatch Extension Functions.

This code overides the core simulation functions in `simulate.py` to allow
simulations to work with RigidBody objects. See `simulate.py` for a detailed
description of the use of single dispatch in simulation functions.

These functions are based on Miller III et al [1], which uses the
Suzuki-Trotter decomposition to identify a factorization of the Liouville
operator for Rigid Body motion. This factorization is compatible with either
the NVE or NVT ensemble (but is not compatible with NPT).
"""

@quantity.count_dof.register
def _(position: RigidBody) -> int:
  sizes = rigid_body.tree_map_no_quat(lambda x: x.size, position)
  return tree_reduce(lambda accum, x: accum + x, sizes, 0)

@simulate.initialize_momenta.register(RigidBody)
def _(state, key: Array, kT: float):
  R, mass = state.position, state.mass
  center_key, angular_key = random.split(key)

  P_center = jnp.sqrt(mass.center * kT) * random.normal(
    center_key, R.center.shape, dtype=R.center.dtype
  )
  P_center = P_center - jnp.mean(P_center, axis=0, keepdims=True)

  # A the moment we assume that rigid body objects are either 2d or 3d. At some
  # point it might be worth expanding this definition to include other kinds of
  # oriented bodies.
  if isinstance(R.orientation, Quaternion):
    scale = jnp.sqrt(mass.orientation * kT)
    center = R.center
    P_angular = scale * random.normal(
      angular_key, center.shape, dtype=center.dtype
    )
    P_orientation = rigid_body.angular_momentum_to_conjugate_momentum(
      R.orientation, P_angular
    )
  else:
    scale = jnp.sqrt(mass.orientation * kT)
    shape, dtype = R.orientation.shape, R.orientation.dtype
    P_orientation = scale * random.normal(angular_key, shape, dtype=dtype)

  return state.set(momentum=RigidBody(P_center, P_orientation))


@simulate.position_step.register(RigidBody)
def _(state, shift_fn, dt, m_rot=1, **kwargs):
  if isinstance(state.position.orientation, Quaternion):
    return rigid_body._rigid_body_3d_position_step(
      state, shift_fn, dt, m_rot=m_rot, **kwargs
    )
  else:
    return rigid_body._rigid_body_2d_position_step(state, shift_fn, dt, **kwargs)


@simulate.stochastic_step.register(RigidBody)
def _(state, dt: float, kT: float, gamma: float):
  key, center_key, orientation_key = random.split(state.rng, 3)

  rest, center, orientation = rigid_body.split_center_and_orientation(state)

  center = simulate.stochastic_step(
    center.set(rng=center_key), dt, kT, gamma.center
  )

  Pi = orientation.momentum.vec
  I = orientation.mass
  G = gamma.orientation

  M = 4 / jnp.sum(1 / I, axis=-1)
  Q = orientation.position.vec
  P = MOMENTUM_PERMUTATION

  # First evaluate PI term
  Pi_mean = 0
  for l in range(3):
    I_l = I[:, [l], None]
    M_l = M[:, None, None]
    PP = P[l](Q)[:, None, :] * P[l](Q)[:, :, None]
    Pi_mean += jnp.exp(-G * M_l * dt / (4 * I_l)) * PP
  Pi_mean = jnp.einsum('nij,nj->ni', Pi_mean, Pi)

  # Then evaluate Q term
  Pi_var = 0
  for l in range(3):
    scale = jnp.sqrt(
      4 * kT * I[:, l] * (1 - jnp.exp(-M * G * dt / (2 * I[:, l])))
    )
    Pi_var += (scale[:, None] * P[l](Q)) ** 2

  momentum_dist = simulate.Normal(Pi_mean, Pi_var)
  new_momentum = rigid_body.Quaternion(momentum_dist.sample(orientation_key))
  orientation = orientation.set(momentum=new_momentum)

  return rigid_body.merge_center_and_orientation(rest.set(rng=key), center, orientation)


@simulate.canonicalize_mass.register(RigidBody)
def _(state):
  mass = state.mass
  if len(mass.center) == 1:
    return state.set(mass=RigidBody(mass.center[0], mass.orientation))
  elif len(mass.center) > 1:
    return state.set(mass=RigidBody(mass.center[:, None], mass.orientation))
  raise NotImplementedError(
    'Center of mass must be either a scalar or a vector. Found an array of '
    f'shape {mass.center.shape}.'
  )


@simulate.kinetic_energy.register(RigidBody)
def _(state) -> Array:
  return rigid_body.kinetic_energy(state.position, state.momentum, state.mass)


@simulate.temperature.register(RigidBody)
def _(state) -> Array:
  return rigid_body.temperature(state.position, state.momentum, state.mass)
