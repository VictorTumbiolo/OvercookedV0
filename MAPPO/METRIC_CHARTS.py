import matplotlib.pyplot as plt
import numpy as np

# data
layouts = ["Cramped Room", "Coordination Ring", "Counter Circuit"]

# Data complied into and pulled from metrics_compiled.txt
optimal_potting = [36.00, 39.01, 29.54]
soup_pickups = [11.00, 11.99, 8.14]
soups = [11.00, 11.88, 7.99]

# Colors per layout
colors = ["green", "orange", "blue"]

x = np.arange(len(layouts))
width = 0.35


# grapg 1 Absolute Metrics
plt.figure()

plt.bar(x - width/2, optimal_potting, width, label="Optimal Potting", color=colors)
plt.bar(x + width/2, soup_pickups, width, label="Soup Pickups", color=colors, alpha=0.6)

plt.xlabel("Layout")
plt.ylabel("Average Count")
plt.title("Coordination and Efficiency Metrics Across Layouts")
plt.xticks(x, layouts)
plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("absolute_metrics.png", dpi=300)
plt.show()

# graph 2 normalixzed Metrics

# Compute ratios
potting_per_soup = [p/s for p, s in zip(optimal_potting, soups)]
pickup_per_soup = [p/s for p, s in zip(soup_pickups, soups)]

plt.figure()

plt.bar(x - width/2, potting_per_soup, width, label="Potting per Soup", color=colors)
plt.bar(x + width/2, pickup_per_soup, width, label="Pickups per Soup", color=colors, alpha=0.6)

plt.xlabel("Layout")
plt.ylabel("Ratio")
plt.title("Normalized Coordination and Efficiency Metrics")
plt.xticks(x, layouts)
plt.legend()
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("normalized_metrics.png", dpi=300)
plt.show()