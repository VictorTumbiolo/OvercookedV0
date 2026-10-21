import matplotlib.pyplot as plt

files = [
    ("logs_PPO/circuit_ppo_training_log.txt", "Counter Circuit"),
    ("logs_PPO/ring_ppo_training_log.txt", "Coordination Ring"),
    ("logs_PPO/cramped_ppo_training_log.txt", "Cramped Room"),
]

plt.figure(figsize=(8, 5))

for file_path, label in files:
    episodes = []
    running_avg = []

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        parts = line.split()

        if len(parts) < 4:
            continue

        episode = float(parts[0])
        runavg = float(parts[3])

        episodes.append(episode)
        running_avg.append(runavg)

    # Plot running average
    plt.plot(episodes, running_avg, label=label)

    #  Find and mark peak
    if running_avg:
        max_val = max(running_avg)
        max_idx = running_avg.index(max_val)
        max_episode = episodes[max_idx]

        plt.scatter(
            max_episode,
            max_val,
            s=60,
            edgecolors='black',
            linewidths=1.5,
            zorder=3
        )

        # plt.text(max_episode, max_val, f"  {label} peak", fontsize=8)

# Labels & styling
plt.xlabel("Episode")
plt.ylabel("Soups (Running 100 Window Average)")
plt.title("Running Average Soups Across Layouts")

plt.legend()
plt.grid()

plt.savefig("ppo_soup_running_avg.png", dpi=300, bbox_inches="tight")
plt.show()