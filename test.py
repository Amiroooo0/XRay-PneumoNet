import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data_audit.csv")
df = df[df["status"] == "ok"].copy()

colors = {"NORMAL": "teal", "PNEUMONIA": "tomato"}
markers = {"train": "o", "val": "^", "test": "s"}

fig, ax = plt.subplots(figsize=(9, 6))

for (label, split), group in df.groupby(["label", "split"]):
    ax.scatter(
        group["width"],
        group["height"],
        color=colors[label],
        marker=markers[split],
        alpha=0.55,
        s=10,
        label=f"{label.title()} · {split}",
    )

ax.set(
    title="Chest X-ray image dimensions",
    xlabel="Width (pixels)",
    ylabel="Height (pixels)",
)
ax.grid(alpha=0.25)
ax.legend()
fig.tight_layout()
plt.show()