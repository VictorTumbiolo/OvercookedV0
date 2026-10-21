# RUN THIS SCRIPT TO VISUALLY WATCH A TRAINED MODEL ON A LAYOUT
import torch
import numpy as np
import pygame

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action

from model import SharedActorCritic

# config options

# MODEL_PATH = "models/cramped_room_best_model.pth"
MODEL_PATH = "models/coordination_ring_best_model.pth"
# MODEL_PATH = "models/counter_circuit_best_model.pth"

# Layouts for TA convience
# LAYOUT = "cramped_room"
LAYOUT = "coordination_ring"
# LAYOUT = "counter_circuit_o_1order"
HORIZON = 400

# TRAINED AGENT
class TrainedAgent(Agent):
    def __init__(self, model_path, mdp, agent_index):
        super().__init__()
        self.device = torch.device("cpu")
        self.agent_index = agent_index


        self.env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

        # Model
        self.model = SharedActorCritic(num_actions=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def action(self, state):
        obs = self.env.featurize_state_mdp(state)
        obs_tensor = torch.from_numpy(np.array(obs)).float().to(self.device)

        with torch.no_grad():
            logits, _ = self.model(obs_tensor)

        # Create a dist based on the logits
        dist = torch.distributions.Categorical(logits=logits)
        # sample dist
        actions = dist.sample()

        action_idx = int(actions[self.agent_index].item())
        return Action.INDEX_TO_ACTION[action_idx], {}


# visual
def run_pygame_playback(trajs, mdp):
    states = trajs['ep_states'][0]
    pygame.init()
    viz = StateVisualizer()
    grid = mdp.terrain_mtx

    first_frame = viz.render_state(states[0], grid)
    width, height = first_frame.get_size()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Overcooked AI Playback")
    clock = pygame.time.Clock()

    for state in states:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

        frame = viz.render_state(state, grid)
        screen.blit(frame, (0, 0))
        pygame.display.flip()
        clock.tick(15)

    pygame.time.wait(1000)
    pygame.quit()


# main
if __name__ == "__main__":
    # hear as a formality, not used for testing
    mdp_params = {
        "layout_name": LAYOUT,
        "rew_shaping_params": {
            "PLACEMENT_IN_POT_REW": 5,
            "DISH_PICKUP_REWARD": 2,
            "SOUP_PICKUP_REWARD": 6
        },
        "old_dynamics": True
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

    print("Simulating episode...")
    trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)

    total_sparse_reward = trajs['ep_returns'][0]
    soups = int(total_sparse_reward / 20)

    print("\n" + "="*30)
    print(f"Total Reward: {total_sparse_reward}")
    print(f"Soups Delivered: {soups}")
    print("="*30 + "\n")

    run_pygame_playback(trajs, mdp)
