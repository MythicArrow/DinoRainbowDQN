from RainBDqn import RainbowDqn
import gym
import jax
import optax
from flax import nnx
def train_dino():
    # Initialize Dino environment
    env = gym.make("ChromeDino-v0")
    state_shape = env.observation_space.shape
    action_dim = env.action_space.n

    # Replay Buffer and Model
    buffer_capacity = 100_000
    replay_buffer = PrioritizedReplayBuffer(buffer_capacity, state_shape, action_dim)
    model = RainbowDQN(action_dim=action_dim, atoms=51, v_min=-10, v_max=10)
    params = nnx.Collection(model.init(jax.random.PRNGKey(0), jnp.ones(state_shape)))
    opt_state = optax.adam(1e-4).init(params)

    # Training parameters
    episodes = 1000
    batch_size = 32
    gamma = 0.99
    rng = jax.random.PRNGKey(0)

    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            # Select action (epsilon-greedy)
            q_values = model.q_values(jnp.expand_dims(state, axis=0))
            action = q_values.argmax() if np.random.rand() > 0.1 else env.action_space.sample()

            # Step environment
            next_state, reward, done, _ = env.step(action)
            replay_buffer.store(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            # Train model
            if replay_buffer.size > batch_size:
                params, opt_state, loss = train_step(rng, params, opt_state, replay_buffer, model, batch_size, gamma)

        print(f"Episode {episode + 1}, Reward: {episode_reward}")
    
    
