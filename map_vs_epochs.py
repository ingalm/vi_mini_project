import pandas as pd
import matplotlib.pyplot as plt

results_csv_path = "runs/detect/train40/results.csv"


# Load the CSV file
results_df = pd.read_csv(results_csv_path)

# Plot mAP@0.5:0.95
plt.figure(figsize=(10, 6))
plt.plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], marker='o', label="mAP@0.5:0.95")
plt.xlabel("Epoch")
plt.ylabel("mAP@0.5:0.95")
plt.title("Evolution of mAP@0.5:0.95 Over Epochs")
plt.grid(True)
plt.legend()
plt.show()
plt.savefig('mAP@0.5:0.95.png')
