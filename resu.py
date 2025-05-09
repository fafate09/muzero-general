import numpy as np

data = np.load("results_muzero.npy", allow_pickle=True)
if isinstance(data, np.ndarray):
    data = data.tolist()

for run_id, run in enumerate(data):
    print(f"--- Run {run_id+1} ---")
    if "coherence_reward_trace" in run:
        for i, (coh, rew) in enumerate(run["coherence_reward_trace"]):
            print(f"  Étape {i+1}: Coherence = {coh:.4f}, Reward = {rew:.4f}")
    else:
        print("  ➤ Aucune trace de cohérence/récompense trouvée.")
