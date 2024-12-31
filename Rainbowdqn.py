import jax
import jax.numpy as jnp
from flax import nnx
import numpy as np
import optax
from typing import Any

# Rainbow DQN Model
class RainbowDQN(nnx.Module):
    action_dim: int
    atoms: int
    v_min: float
    v_max: float

    def __init__(self):
        self.support = jnp.linspace(self.v_min, self.v_max, self.atoms)
        self.feature_layer = nnx.Linear(128)
        self.value_layer = nnx.Sequential([
            nnx.Linear(128), nnx.relu, nnx.Linear(self.atoms)
        ])
        self.advantage_layer = nnx.Sequential([
            nnx.Linear(128), nnx.relu, nnx.Linear(self.action_dim * self.atoms)
        ])

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        features = nnx.relu(self.feature_layer(x))
        value = self.value_layer(features).reshape(-1, 1, self.atoms)
        advantage = self.advantage_layer(features).reshape(-1, self.action_dim, self.atoms)
        q_atoms = value + (advantage - advantage.mean(axis=1, keepdims=True))
        return nnx.softmax(q_atoms, axis=-1)

    def q_values(self, x: jnp.ndarray) -> jnp.ndarray:
        q_atoms = self(x)
        return (q_atoms * self.support).sum(axis=-1)

# Replay Buffer with PER
class PrioritizedReplayBuffer:
    def __init__(self, capacity, state_shape, action_dim, alpha=0.6):
        self.capacity = capacity
        self.ptr, self.size = 0, 0
        self.alpha = alpha
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.priorities = np.zeros(capacity, dtype=np.float32)

    def store(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        self.priorities[self.ptr] = max_priority
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities[:self.size] ** self.alpha
        probabilities = priorities / priorities.sum()
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        weights = (self.size * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return (
            self.states[indices], self.actions[indices], self.rewards[indices],
            self.next_states[indices], self.dones[indices], indices, weights
        )

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

# Loss Function
def distributional_loss(params, target_dist, states, actions, model, support, gamma, dones):
    q_atoms = model.apply(params, states)
    q_atoms = q_atoms[jnp.arange(q_atoms.shape[0]), actions]
    delta_z = (support[-1] - support[0]) / (len(support) - 1)
    target_z = jnp.clip(target_dist + gamma * (1 - dones[:, None]) * support, support[0], support[-1])
    b = (target_z - support[0]) / delta_z
    l, u = jnp.floor(b).astype(jnp.int32), jnp.ceil(b).astype(jnp.int32)
    proj_target = jnp.zeros_like(q_atoms)
    proj_target = proj_target.at[jnp.arange(target_dist.shape[0]), l].add(target_dist * (u - b))
    proj_target = proj_target.at[jnp.arange(target_dist.shape[0]), u].add(target_dist * (b - l))
    return -(proj_target * jnp.log(q_atoms + 1e-5)).sum(axis=-1).mean()

# Training Step
def train_step(rng, params, opt_state, replay_buffer, model, batch_size, gamma=0.99, beta=0.4):
    batch = replay_buffer.sample(batch_size, beta)
    states, actions, rewards, next_states, dones, indices, weights = batch
    next_q_atoms = model.apply(params, next_states)
    next_q = (next_q_atoms * model.support).sum(axis=-1)
    next_actions = next_q.argmax(axis=-1)
    target_dist = model.apply(params, next_states)[jnp.arange(batch_size), next_actions]
    loss_fn = lambda p: distributional_loss(p, target_dist, states, actions, model, model.support, gamma, dones)
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optax.adam(1e-4).update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    td_errors = jnp.abs(loss)  # Approximate TD-errors
    replay_buffer.update_priorities(indices, td_errors)
    return new_params, opt_state, loss

# Example Usage
'''buffer_capacity = 100_000
state_shape, action_dim, atoms, v_min, v_max = (4,), 3, 51, -10, 10
replay_buffer = PrioritizedReplayBuffer(buffer_capacity, state_shape, action_dim)
model = RainbowDQN(action_dim=action_dim, atoms=atoms, v_min=v_min, v_max=v_max)
params = nnx.Collection(model.init(jax.random.PRNGKey(0), jnp.ones((1, 4))))
opt_state = optax.adam(1e-4).init(params)
'''