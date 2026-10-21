import matplotlib.pyplot as plt

# File paths
files = {
    "Counter Circuit": "logs_PPO(COPY)/ppo_circuit_episode_metrics.txt",
    "Cramped Room": "logs_PPO(COPY)/ppo_cramped_episode_metrics.txt",
    "Coordination Ring": "logs_PPO(COPY)/ppo_ring_episode_metrics.txt",
}

plt.figure()

# Loop through each file
for label, file_path in files.items():
    episodes = []
    soups = []

    with open(file_path, "r") as f:
        lines = f.readlines()

        # Skip header
        for line in lines[1:]:
            parts = line.strip().split()

            episode = int(parts[0])
            soup = int(parts[1])  # soups column

            episodes.append(episode)
            soups.append(soup)

    # Plot each layout
    plt.plot(episodes, soups, label=label)

# Labels and formatting
plt.xlabel("Episode")
plt.ylabel("Soups Delivered")
plt.title("Soups per Episode Across Layouts")
plt.legend(loc='lower left', bbox_to_anchor=(0, 0.05))
plt.grid()

plt.savefig("ppo_testing_soups_plot.png", dpi=300, bbox_inches="tight")
plt.show()