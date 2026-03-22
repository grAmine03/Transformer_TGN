import pickle
import numpy as np

test_ap_list = []
new_node_ap_list = []
epoch_time_means = []

# ---- 1) Charger tgn-attn.pkl (run 0) ----
base = "results/dyrep_rnn.pkl"
with open(base, 'rb') as f:
    data = pickle.load(f)
    test_ap_list.append(data["test_ap"])
    new_node_ap_list.append(data["new_node_test_ap"])
    epoch_time_means.append(np.mean(data["epoch_times"]))
    print(f"[OK] {base} -> test_ap={data['test_ap']}, new_node_test_ap={data['new_node_test_ap']}, mean_epoch={np.mean(data['epoch_times']):.4f}")

# ---- 2) Charger tgn-attn_1.pkl à tgn-attn_9.pkl ----
for i in range(1, 10):
    file = f"results/dyrep_rnn_{i}.pkl"
    try:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            test_ap_list.append(data["test_ap"])
            new_node_ap_list.append(data["new_node_test_ap"])
            epoch_time_means.append(np.mean(data["epoch_times"]))

            print(f"[OK] {file} -> test_ap={data['test_ap']}, new_node_test_ap={data['new_node_test_ap']}, mean_epoch={np.mean(data['epoch_times']):.4f}")
    except Exception as e:
        print(f"[ERROR] {file}: {e}")

print("\n==================== FINAL RESULTS ====================\n")
print(f"Test_AP        → Mean = {np.mean(test_ap_list):.4f}, Std = {np.std(test_ap_list):.4f}")
print(f"New_Node_AP    → Mean = {np.mean(new_node_ap_list):.4f}, Std = {np.std(new_node_ap_list):.4f}")
print(f"Epoch_times    → Mean = {np.mean(epoch_time_means):.4f}")