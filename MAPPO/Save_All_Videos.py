import os

import pygame

from overcooked_ai_py.static import LAYOUTS_DIR
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer


LAYOUTS = sorted(
    f[:-len(".layout")] for f in os.listdir(LAYOUTS_DIR) if f.endswith(".layout")
)

OUT_DIR = "layout_images"


if __name__ == "__main__":
    pygame.init()
    os.makedirs(OUT_DIR, exist_ok=True)

    viz = StateVisualizer()

    for layout in LAYOUTS:
        try:
            mdp = OvercookedGridworld.from_layout_name(layout)
            state = mdp.get_standard_start_state()

            # the frame is sized from the layout's grid, so each image fits its own map
            frame = viz.render_state(state, mdp.terrain_mtx)
        except Exception as e:
            print(f"Skipped {layout}: {e}")
            continue

        path = os.path.join(OUT_DIR, f"{layout}.png")
        pygame.image.save(frame, path)

        width, height = frame.get_size()
        print(f"Saved {path} ({width}x{height})")

    pygame.quit()
