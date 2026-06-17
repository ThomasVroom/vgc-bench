import os
import pickle

import torch
import pandas as pd
from tqdm import tqdm
from imitation.data.types import Trajectory

from vgc_bench.src.utils import load_policy
from vgc_bench.src.policy import MaskedActorCriticPolicy


def convert_trajs(trajs_dir: str, policies: list[str], device: torch.device = torch.device("cuda:0")):
    agents = [load_policy(MaskedActorCriticPolicy, p, device).eval() for p in policies]
    vecs = [[] for _ in range(len(agents))]
    vals = [[] for _ in range(len(agents))]

    data = []
    for id, file in enumerate(tqdm(os.listdir(trajs_dir))):
        # load trajectory
        with open(f"{trajs_dir}/{file}",'rb') as f:
            traj: Trajectory = pickle.load(f)
        winner = bool(traj.infos[0]) if traj.infos is not None else True

        # convert each batch of states to encoded latent vectors
        obs = dict(observation=torch.from_numpy(traj.obs).to(device))
        for i in range(len(agents)):
            vecs[i] = agents[i].features_extractor(obs).detach().cpu().numpy()
            vals[i] = agents[i].predict_values(obs).detach().cpu() # type: ignore
        for i in range(len(traj.obs)):
            data.append(dict(battle_id=id, depth=i, winner=winner))
            for j in range(len(agents)):
                data[-1][f"encoding_{j}"] = vecs[j][i]
                data[-1][f"value_{j}"] = vals[j][i].item()

    # save encodings
    df = pd.DataFrame(data)
    print("saving encodings...")
    df.to_pickle(f"{trajs_dir}_enc.pkl")


if __name__ == "__main__":
    # Reg F
    convert_trajs("trajs/gen9vgc2024regf", [
        "results/saves_sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_bc-sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_i/64_teams/seed1/5013504.zip",
        "results/saves_bc-sp/reg_i/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_f_to_i/64_teams/seed1/10027008.zip"
    ])

    # Reg I
    convert_trajs("trajs/gen9vgc2025regi", [
        "results/saves_sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_bc-sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_i/64_teams/seed1/5013504.zip",
        "results/saves_bc-sp/reg_i/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_f_to_i/64_teams/seed1/10027008.zip"
    ])

    # dev = torch.device("cpu")
    # agent_a = load_policy(MaskedActorCriticPolicy, "results_temp/saves_sp/1_columns/reg_a/64_teams/seed1/98304.zip", dev)
    # agent_b = load_policy(MaskedActorCriticPolicy, "results_temp/saves_sp/2_columns/reg_a_to_b/64_teams/seed1/98304.zip", dev)
    # for (n_a, p_a), (n_b, p_b) in zip(
    #     agent_a.features_extractor.column_head.named_parameters(),
    #     agent_b.features_extractor.column_head.prev_column.named_parameters()
    # ):
    #     print(n_a, p_a.equal(p_b))
