import pickle

import torch
import pandas as pd
from tqdm import tqdm

from vgc_bench.src.utils import load_policy
from vgc_bench.src.policy import MaskedActorCriticPolicy

def convert_trajs(trajs_file: str, policies: list[str], device: torch.device = torch.device("cuda:0")):
    agents = [load_policy(MaskedActorCriticPolicy, p, device).eval() for p in policies]
    data = []

    with open(trajs_file, 'rb') as f:
        trajs_df: pd.DataFrame = pickle.load(f)

    with torch.no_grad():
        for _, row in tqdm(trajs_df.iterrows()):
            obs_dict = {
                "observation": torch.from_numpy(row["obs"]).unsqueeze(0).to(device=device),
                "action_mask": torch.from_numpy(row["mask"]).unsqueeze(0).to(device=device),
            }
            data.append(dict(depth=row["depth"]))
            for i, policy in enumerate(agents):
                _, values, _ = policy.forward(obs_dict, deterministic=True)
                action_dist1 = policy.temp_distribution.distribution[0].probs[0] # type: ignore
                action_dist2 = policy.temp_distribution.distribution[1].probs[0] # type: ignore

                data[-1][f"value_{i}"] = values.item()
                data[-1][f"action1_{i}"] = action_dist1.detach().cpu().numpy()
                data[-1][f"action2_{i}"] = action_dist2.detach().cpu().numpy()

                logits = policy.features_extractor(obs_dict)
                data[-1][f"logits_{i}"] = logits.detach().cpu().numpy()

    df = pd.DataFrame(data)
    print("saving encodings...")
    df.to_pickle(f"{trajs_file[:-4]}_enc1.pkl")

if __name__ == "__main__":
    # Reg F
    convert_trajs("trajs/regf_trajs_indist.pkl", [
        "results_nonorm/saves_sp/1_columns/reg_f/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/5_columns/reg_a_to_b_to_c_to_d_to_e/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/6_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/5013504.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e/64_teams/seed1/25067520.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/30081024.zip"
    ])
    convert_trajs("trajs/regf_trajs_outofdist.pkl", [
        "results_nonorm/saves_sp/1_columns/reg_f/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/5_columns/reg_a_to_b_to_c_to_d_to_e/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/6_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/5013504.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e/64_teams/seed1/25067520.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/30081024.zip"
    ])

    # Reg G
    convert_trajs("trajs/regg_trajs_indist.pkl", [
        "results_nonorm/saves_sp/1_columns/reg_g/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/6_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/7_columns/reg_a_to_b_to_c_to_d_to_e_to_f_to_g/64_teams/seed1/5013504.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/30081024.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f_to_g/64_teams/seed1/35094528.zip"
    ])
    convert_trajs("trajs/regg_trajs_outofdist.pkl", [
        "results_nonorm/saves_sp/1_columns/reg_g/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/6_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/5013504.zip",
        "results_nonorm/saves_sp/7_columns/reg_a_to_b_to_c_to_d_to_e_to_f_to_g/64_teams/seed1/5013504.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f/64_teams/seed1/30081024.zip",
        "results_l2/saves_sp/1_columns/reg_a_to_b_to_c_to_d_to_e_to_f_to_g/64_teams/seed1/35094528.zip"
    ])

    # # test column freezing
    # dev = torch.device("cpu")
    # agent_a = load_policy(MaskedActorCriticPolicy, "results_temp/saves_sp/1_columns/reg_a/64_teams/seed1/98304.zip", dev)
    # agent_b = load_policy(MaskedActorCriticPolicy, "results_temp/saves_sp/2_columns/reg_a_to_b/64_teams/seed1/98304.zip", dev)
    # for (n_a, p_a), (n_b, p_b) in zip(
    #     agent_a.features_extractor.column_head.named_parameters(),
    #     agent_b.features_extractor.column_head.prev_column.named_parameters()
    # ):
    #     print(n_a, p_a.equal(p_b))
