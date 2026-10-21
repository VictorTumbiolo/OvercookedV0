import torch
import numpy as np
import sys
import os
import gym

# scripts fro mdifferent directories
sys.path.append(os.getcwd())

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from model import SharedActorCritic
from PPO_Agent import PPOUpdater

# Save for testing
def save_model(model, current_step):
    path = f"cramped/{current_step}_best_model.pth"
    os.makedirs("cramped", exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    # setup
    layout = "cramped_room"
    max_steps = 5000000
    update_every = 400
    device = torch.device("cpu")

    reward_shaping = {
        "PLACEMENT_IN_POT_REW": 5,
        "DISH_PICKUP_REWARD": 2,
        "SOUP_PICKUP_REWARD": 6
    }

    mdp = OvercookedGridworld.from_layout_name(
        layout,
        rew_shaping_params=reward_shaping,
        old_dynamics=True
    )

    base_env = OvercookedEnv.from_mdp(mdp, horizon=400)


    env = gym.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp
    )

    # model
    obs = env.reset()["both_agent_obs"]  # (2, 96)

    model = SharedActorCritic(num_actions=6).to(device)
    ppo = PPOUpdater(model, lr=3e-4, clip_val=0.2)

    # tracking info
    current_step = 0
    ep_reward_tracker = []
    soups_in_batch = 0
    ep_count_in_batch = 0

    soups_this_episode = 0
    soups_per_episode = []

    best_running_avg = -float("inf")

    print(f"Starting training on {layout}...")

    # log info
    os.makedirs("logs", exist_ok=True)
    log_path = "logs/cramped_ppo_training_log.txt"

    log_file = open(log_path, "w")
    log_file.write(f"{'Episode':<8}{'Reward':<8}{'Soups':<6}{'RunAvg':<5}\n")

    # Training
    while current_step < max_steps:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            logits, values = model(obs_tensor)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        action_list = [int(a) for a in actions.numpy()]

        # step handling
        next_obs_dict, sparse_reward, done, info = env.step(action_list)
        next_obs = next_obs_dict["both_agent_obs"]

        # Convert to per-agent rewards
        rewards = [sparse_reward, sparse_reward]

        # Add shaping
        shaped = info.get('shaped_r_by_agent', [0, 0])

        for i in range(2):
            rewards[i] += shaped[i]
            rewards[i] -= 0.001

        # track soups
        sparse_rew = sparse_reward
        if sparse_rew > 0:
            soups = sparse_rew // 20
            soups_in_batch += soups
            soups_this_episode += soups

        # store transitons
        for i in range(2):
            ppo.buffer.store(
                obs[i].copy(),
                actions[i].view(1),
                rewards[i],
                log_probs[i].view(1),
                values[i].view(1),
                done
            )

        obs = next_obs
        current_step += 1
        ep_reward_tracker.append(sum(rewards))

        # update
        if current_step % update_every == 0:
            ppo.update()

            avg_rew = np.mean(ep_reward_tracker) if ep_reward_tracker else 0
            avg_soups = soups_in_batch / ep_count_in_batch if ep_count_in_batch > 0 else 0

            if len(soups_per_episode) >= 100:
                running_avg_soups = np.mean(soups_per_episode[-100:])
            else:
                running_avg_soups = np.mean(soups_per_episode) if soups_per_episode else 0

            print(f"Step: {current_step} | Avg Reward: {avg_rew:.3f} | Batch Soups/Ep: {avg_soups:.2f} | Running(100): {running_avg_soups:.2f}")

            # log info to txt file
            log_file.write(
                f"{current_step / update_every:<8}{avg_rew:<8.4f}{avg_soups:<6.2f}{running_avg_soups:<5.2f}\n")
            log_file.flush()

            if running_avg_soups > best_running_avg:
                best_running_avg = running_avg_soups
                save_model(model, current_step)
                print(f"New best model saved! Running Avg: {best_running_avg:.2f}")

            ep_reward_tracker, soups_in_batch, ep_count_in_batch = [], 0, 0

        # reset episode
        if done:
            ep_count_in_batch += 1
            soups_per_episode.append(soups_this_episode)
            soups_this_episode = 0
            obs = env.reset()["both_agent_obs"]

    print("Training complete.")
    log_file.close()


if __name__ == "__main__":
    main()
