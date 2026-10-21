import torch
import numpy as np

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.mdp.actions import Action

from model import SharedActorCritic


# config info
MODEL_PATH = "circuit_temp/3413600_best_model.pth"
LAYOUT = "counter_circuit_o_1order"
HORIZON = 400
NUM_GAMES = 100

EPISODE_FILE = "testing_logs/counter_circuit_episode_metrics.txt"
SUMMARY_FILE = "testing_logs/counter_circuit_summary.txt"

# trained MAPPO agent
class TrainedAgent(Agent):
    def __init__(self, model_path, mdp, agent_index):
        super().__init__()
        # No need for GPU as mentioned in assigment
        self.device = torch.device("cpu")
        self.agent_index = agent_index

        self.env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

        self.model = SharedActorCritic(input_dim=96, num_actions=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def action(self, state):
        both_obs = self.env.featurize_state_mdp(state)
        obs_tensor = torch.tensor(both_obs, dtype=torch.float32).to(self.device)

        local_x = obs_tensor[self.agent_index].unsqueeze(0)
        global_state = torch.cat([obs_tensor[0], obs_tensor[1]]).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(local_x, global_state)

        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample().item()

        return Action.INDEX_TO_ACTION[action_idx], {}


# main
if __name__ == "__main__":
    # For formallity
    mdp_params = {
        "layout_name": LAYOUT,
        "rew_shaping_params": {
            "PLACEMENT_IN_POT_REW": 5,
            "DISH_PICKUP_REWARD": 2,
            "SOUP_PICKUP_REWARD": 6
        }
    }

    print(f"Initializing Evaluator for {LAYOUT}...")

    ae = AgentEvaluator.from_layout_name(
        mdp_params=mdp_params,
        env_params={"horizon": HORIZON}
    )

    mdp = ae.env.mdp

    agent0 = TrainedAgent(MODEL_PATH, mdp, agent_index=0)
    agent1 = TrainedAgent(MODEL_PATH, mdp, agent_index=1)
    agent_pair = AgentPair(agent0, agent1)

    print(f"\nRunning {NUM_GAMES} evaluation episodes...\n")

    trajs = ae.evaluate_agent_pair(agent_pair, num_games=NUM_GAMES)


    # first file
    # Per episode data
    with open(EPISODE_FILE, "w") as f:
        f.write("Episode Soups OptimalPotting SoupPickups SoupDrops\n")

        total_soups = 0
        total_optimal = 0
        total_drops = 0
        total_pickups = 0

        soups_per_game = []

        # get info
        for i in range(NUM_GAMES):
            total_sparse_reward = trajs['ep_returns'][i]
            soups = int(total_sparse_reward / 20)

            soups_per_game.append(soups)
            total_soups += soups

            ep_info = trajs['ep_infos'][i]
            final_step = ep_info[-1]
            game_stats = final_step['episode']['ep_game_stats']

            # optimal Potting
            opt_data = game_stats['optimal_onion_potting']
            optimal = len(opt_data[0]) + len(opt_data[1])
            total_optimal += optimal

            # soup drops
            drop_data = game_stats['soup_drop']
            drops = len(drop_data[0]) + len(drop_data[1])
            total_drops += drops

            # soup pickups
            pickup_data = game_stats['soup_pickup']
            pickups = len(pickup_data[0]) + len(pickup_data[1])
            total_pickups += pickups

            f.write(f"{i+1} {soups} {optimal} {pickups} {drops}\n")


    # sumary stats
    avg_soups = total_soups / NUM_GAMES
    avg_optimal = total_optimal / NUM_GAMES
    avg_drops = total_drops / NUM_GAMES
    avg_pickups = total_pickups / NUM_GAMES

    std_soups = np.std(soups_per_game)
    min_soups = np.min(soups_per_game)
    max_soups = np.max(soups_per_game)


    # second file
    # summary stat info
    with open(SUMMARY_FILE, "w") as f:
        f.write("===== FINAL RESULTS (100 GAMES) =====\n")
        f.write(f"Average Soups Delivered: {avg_soups:.2f}\n")
        f.write(f"Average Optimal Potting: {avg_optimal:.2f}\n")
        f.write(f"Average Soup Pickups: {avg_pickups:.2f}\n")
        f.write(f"Average Soup Drops: {avg_drops:.2f}\n\n")

        f.write("--- Totals ---\n")
        f.write(f"Total Soups: {total_soups}\n")
        f.write(f"Total Optimal Potting: {total_optimal}\n")
        f.write(f"Total Pickups: {total_pickups}\n")
        f.write(f"Total Drops: {total_drops}\n\n")

        f.write("--- Variability ---\n")
        f.write(f"Std Dev Soups: {std_soups:.2f}\n")
        f.write(f"Min Soups: {min_soups}\n")
        f.write(f"Max Soups: {max_soups}\n")
        f.write("====================================\n")

    print(f"Saved episode data to: {EPISODE_FILE}")
    print(f"Saved summary data to: {SUMMARY_FILE}")