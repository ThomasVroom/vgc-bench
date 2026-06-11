import io
import torch
import zipfile
import numpy as np

regs = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"]
run_id = 1
source_results_suffix = ""
source_suffix = f"_{source_results_suffix}" if source_results_suffix else ""
suffix = "results" + source_suffix + "/saves_sp"

# actor
pi_proj_alphas = []
pi_transformer_alphas = []

# critic
vf_proj_alphas = []
vf_transformer_alphas = []

for i in range(1, 10):
    path = suffix + f"/{i+1}_columns/reg_{'_to_'.join(regs[:(i+1)])}/64_teams/seed{run_id}/5013504.zip"
    with zipfile.ZipFile(path, "r") as zf:
        with zf.open("policy.pth") as f:
            state_dict: dict = torch.load(io.BytesIO(f.read()), map_location="cpu", weights_only=True)
    pi_proj_alphas.append(state_dict["pi_features_extractor.column_head.pokemon_proj_alpha"].item())
    pi_transformer_alphas.append(state_dict["pi_features_extractor.column_head.transformer_alphas"].numpy())
    vf_proj_alphas.append(state_dict["vf_features_extractor.column_head.pokemon_proj_alpha"].item())
    vf_transformer_alphas.append(state_dict["vf_features_extractor.column_head.transformer_alphas"].numpy())

print("- ACTOR -")
pi_proj_alphas = np.array(pi_proj_alphas)
print("proj_alphas:", pi_proj_alphas)
pi_transformer_alphas = np.array(pi_transformer_alphas)
print("transformer_alphas:", pi_transformer_alphas)
print()

print("- CRITIC -")
vf_proj_alphas = np.array(vf_proj_alphas)
print("proj_alphas:", vf_proj_alphas)
vf_transformer_alphas = np.array(vf_transformer_alphas)
print("transformer_alphas:", vf_transformer_alphas)
