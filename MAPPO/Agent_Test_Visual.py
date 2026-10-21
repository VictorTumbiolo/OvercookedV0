import torch
import numpy as np
import pygame
import imageio

from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.agents.agent import Agent, AgentPair
from overcooked_ai_py.agents.benchmarking import AgentEvaluator
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.actions import Action

from model import SharedActorCritic


# config info

# MODEL_PATH = "models/cramped_best_model.pth"
# LAYOUT = "cramped_room"

MODEL_PATH = "models/counter_circuit_best_model.pth"
LAYOUT = "counter_circuit_o_1order"

# MODEL_PATH = "models/counter_circuit_best_model.pth"
# LAYOUT = "counter_circuit_o_1order"


HORIZON = 400
VIDEO_PATH = f"MAPPO_{LAYOUT}_playback.mp4"



# trained MAPPO agent
class TrainedAgent(Agent):
    def __init__(self, model_path, mdp, agent_index):
        super().__init__()
        self.device = torch.device("cpu")
        self.agent_index = agent_index

        self.env = OvercookedEnv.from_mdp(mdp, horizon=HORIZON)

        # include input_dim
        self.model = SharedActorCritic(input_dim=96, num_actions=6).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def action(self, state):

        both_obs = self.env.featurize_state_mdp(state)

        obs_tensor = torch.tensor(both_obs, dtype=torch.float32).to(self.device)

        # Local observation for this agent
        local_x = obs_tensor[self.agent_index].unsqueeze(0)

        # Global state (MAPPO critic input)
        global_state = torch.cat([obs_tensor[0], obs_tensor[1]]).unsqueeze(0)

        with torch.no_grad():
            logits, _ = self.model(local_x, global_state)

        dist = torch.distributions.Categorical(logits=logits)
        action_idx = dist.sample().item()

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
    pygame.display.set_caption("MAPPO Overcooked Playback")
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

    pygame.time.wait(1500)
    pygame.quit()



# save the episode as an mp4
def save_mp4(trajs, mdp, path, fps=15):
    states = trajs['ep_states'][0]
    pygame.init()
    viz = StateVisualizer()
    grid = mdp.terrain_mtx

    with imageio.get_writer(path, fps=fps) as writer:
        for state in states:
            frame = viz.render_state(state, grid)
            # pygame arrays are (width, height, 3); video frames are (height, width, 3)
            writer.append_data(np.transpose(pygame.surfarray.array3d(frame), (1, 0, 2)))

    print(f"Saved video to {path}")



# main
if __name__ == "__main__":

    # for formality
    mdp_params = {
        "layout_name": LAYOUT,
        "rew_shaping_params": {
            "PLACEMENT_IN_POT_REW": 3,
            "DISH_PICKUP_REWARD": 3,
            "SOUP_PICKUP_REWARD": 5
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

    print("Simulating episode...")
    trajs = ae.evaluate_agent_pair(agent_pair, num_games=1)

    total_sparse_reward = trajs['ep_returns'][0]
    soups = int(total_sparse_reward / 20)

    print("\n" + "=" * 30)
    print(f"Total Reward: {total_sparse_reward}")
    print(f"Soups Delivered: {soups}")
    print("=" * 30 + "\n")

    save_mp4(trajs, mdp, VIDEO_PATH)

    run_pygame_playback(trajs, mdp)
