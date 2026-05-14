# MONTE CARLO CONTROL ALGORITHM

## AIM
To implement the Monte Carlo Control algorithm and find the optimal policy in a reinforcement learning environment.

## PROBLEM STATEMENT
The problem is to apply the Monte Carlo Control algorithm to the Frozen Lake environment in order to learn an optimal policy for navigating the grid world. In this setting, the agent receives sparse and stochastic rewards, making it challenging to estimate accurate action values. Monte Carlo Control addresses this by sampling complete episodes, estimating the returns for state–action pairs, and iteratively updating the action-value function (Q(s,a)). The learned (Q)-values are then used to improve the policy by balancing exploration and exploitation until convergence. The objective is to demonstrate how Monte Carlo Control can effectively handle the uncertainty of Frozen Lake and produce a policy that maximizes the probability of successfully reaching the goal.

## MONTE CARLO CONTROL ALGORITHM

1. Initialize Q(s, a) arbitrarily for all state-action pairs

2. Initialize returns(s, a) to empty for all state-action pairs

3. Initialize policy π(s) to be arbitrary (e.g., ε-greedy)

4. For each episode:
   a. Generate an episode using policy π
   b. For each state-action pair (s, a) in the episode:
       i. Calculate G (return) for that (s, a) pair
       ii. Append G to returns(s, a)
       iii. Calculate the average of returns(s, a)
       iv. Update Q(s, a) using the average return
    c. Update policy π(s) based on Q(s, a)


## MONTE CARLO CONTROL FUNCTION

```
from tqdm import tqdm
def mc_control(env, gamma=1.0,init_alpha=0.5, min_alpha=0.01, alpha_decay_ratio=0.5,init_epsilon=1.0, min_epsilon=0.1, epsilon_decay_ratio=0.9,n_episodes=3000, max_steps=200, first_visit=True):
    nS, nA = env.observation_space.n, env.action_space.n
    discounts = np.logspace(0, max_steps,num=max_steps,base=gamma,endpoint=False)
    alphas = decay_schedule(init_alpha,min_alpha,alpha_decay_ratio,n_episodes)
    epsilons = decay_schedule(init_epsilon,min_epsilon,epsilon_decay_ratio,n_episodes)
    pi_track = []
    Q = np.zeros((nS, nA), dtype=np.float64)
    Q_track = np.zeros((n_episodes, nS, nA), dtype=np.float64)
    select_action = lambda state, Q, epsilon: (np.argmax(Q[state])
        if np.random.random() > epsilon
        else np.random.randint(len(Q[state]))
    )
    for e in tqdm(range(n_episodes), leave=False):
        trajectory = generate_trajectory(select_action,Q,epsilons[e],env,max_steps)
        visited = np.zeros((nS, nA), dtype=bool)
        for t, (state, action, reward, _, _) in enumerate(trajectory):
            if visited[state][action] and first_visit:
                continue
            visited[state][action] = True
            n_steps = len(trajectory[t:])
            G = np.sum(discounts[:n_steps] * trajectory[t:, 2])
            Q[state][action] = Q[state][action] + alphas[e] * (G - Q[state][action])
        Q_track[e] = Q
        pi_track.append(np.argmax(Q, axis=1))
    V = np.max(Q, axis=1)
    pi = np.argmax(Q, axis=1)
    return Q, V, pi
```

## OUTPUT:

### Name: D Vergin Jenifer
### Register Number: 212223240174

<img width="990" height="333" alt="image" src="https://github.com/user-attachments/assets/a3649468-daf5-4f58-9a2d-2dbe7a5ba0ce" />
<img width="722" height="52" alt="image" src="https://github.com/user-attachments/assets/42ca6da1-4391-4ce5-8f6b-24a10c8c9b8f" />


## RESULT:
Thus, the Monte Carlo Control algorithm was successfully implemented, and the optimal policy and value function for the reinforcement learning environment were obtained.
