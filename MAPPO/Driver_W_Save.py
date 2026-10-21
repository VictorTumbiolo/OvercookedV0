import torch
import numpy as np
import sys
import os
import gym

sys.path.append(os.getcwd())

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv

from model import SharedActorCritic
from PPO_Agent import PPOUpdater

from overcooked_ai_py.mdp.actions import Action


def save_model(model, current_step):
    path = f"save_directory/{current_step}_best_model.pth"
    os.makedirs("save_directory", exist_ok=True)
    torch.save(model.state_dict(), path)


def main():
    layout = "counter_circuit_o_1order"
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
        rew_shaping_params=reward_shaping
    )

    base_env = OvercookedEnv.from_mdp(mdp, horizon=400, info_level=0)
    # From Assignment notebook insturctions
    env = gym.make(
        "Overcooked-v0",
        base_env=base_env,
        featurize_fn=base_env.featurize_state_mdp
    )

    model = SharedActorCritic(input_dim=96, num_actions=6).to(device)
    ppo = PPOUpdater(model, lr=3e-4, clip_val=0.2)

    obs = env.reset()["both_agent_obs"]

    # Track info
    current_step = 0

    ep_reward_tracker = []
    soups_in_batch = 0
    ep_count_in_batch = 0

    soups_this_episode = 0
    soups_per_episode = []

    best_running_avg = -float("inf")

    print(f"Starting training on {layout}...")

    os.makedirs("logs", exist_ok=True)
    log_path = "logs/circuit_temp_training_log.txt"

    log_file = open(log_path, "w")
    log_file.write(f"{'Episode':<8}{'Reward':<8}{'Soups':<6}{'RunAvg':<5}\n")

    while current_step < max_steps:

        # pick action
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)

            # mappo = global state
            global_state = torch.cat([obs_tensor[0], obs_tensor[1]]).unsqueeze(0)
            global_state = global_state.expand(2, -1)

            logits, values = model(obs_tensor, global_state)

            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)

        action_list = actions.cpu().numpy().tolist()

        # step
        next_obs_dict, sparse_reward, done, info = env.step(action_list)
        next_obs = next_obs_dict["both_agent_obs"]

        # fixed rewards
        rewards = [sparse_reward, sparse_reward]
        shaped = info.get('shaped_r_by_agent', [0, 0])

        # Per Agent Reward
        for i in range(2):
            rewards[i] += shaped[i]
            rewards[i] -= 0.001


        # soup tracking
        if sparse_reward > 0:
            soups = sparse_reward // 20
            soups_in_batch += soups
            soups_this_episode += soups

        # store in buffer
        for i in range(2):
            ppo.buffer.store(
                obs[i].copy(),
                global_state[i].cpu().numpy(),
                actions[i].view(1),
                rewards[i],
                log_probs[i].view(1),
                values[i].view(1),
                done
            )

        obs = next_obs
        current_step += 1
        ep_reward_tracker.append(sum(rewards))

        # update every episode
        if current_step % update_every == 0:
            ppo.update()

            avg_rew = np.mean(ep_reward_tracker) if ep_reward_tracker else 0
            avg_soups = soups_in_batch / ep_count_in_batch if ep_count_in_batch > 0 else 0

            # Track window
            if len(soups_per_episode) >= 100:
                running_avg_soups = np.mean(soups_per_episode[-100:])
            else:
                running_avg_soups = np.mean(soups_per_episode) if soups_per_episode else 0
            # formatting consistency
            print(f"Step: {current_step} | Avg Reward: {avg_rew:.3f} | Batch Soups/Ep: {avg_soups:.2f} | Running(100): {running_avg_soups:.2f}")
            log_file.write(f"{current_step / update_every:<8}{avg_rew:<8.4f}{avg_soups:<6.2f}{running_avg_soups:<5.2f}\n")
            log_file.flush()  # ensures it's written immediately

            if running_avg_soups > best_running_avg:
                best_running_avg = running_avg_soups
                save_model(model, current_step)
                print(f"New best model saved! Running Avg: {best_running_avg:.2f}")

            ep_reward_tracker, soups_in_batch, ep_count_in_batch = [], 0, 0

        # reset
        if done:
            ep_count_in_batch += 1
            soups_per_episode.append(soups_this_episode)
            soups_this_episode = 0
            obs = env.reset()["both_agent_obs"]

    print("Training complete.")
    log_file.close()


if __name__ == "__main__":
    main()
