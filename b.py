import json
import matplotlib.pyplot as plt

# JSONファイルの読み込み
with open("output.json", "r") as f:
    data = json.load(f)


for key in sorted(data.keys(), key=lambda k: int(k)):
    values = data[key]
    x = [i * 10 for i in range(len(values))]  # X軸を10倍
    if key == "0":
        label = "single"
    elif key == "5":
        label = "multi"
    else:
        continue
    plt.plot(x, values, label=label)

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Agent Performance")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
