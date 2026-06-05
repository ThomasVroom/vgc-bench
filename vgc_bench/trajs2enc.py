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
    data = []
    for id, file in enumerate(tqdm(os.listdir(trajs_dir))):
        # load trajectory
        with open(f"{trajs_dir}/{file}",'rb') as f:
            traj: Trajectory = pickle.load(f)
        # convert each batch of states to encoded latent vectors
        obs = dict(observation=torch.from_numpy(traj.obs[:-1]).to(device)) # [:-1] -> skip terminal state
        for i in range(len(agents)):
            vecs = agents[i].features_extractor(obs).detach().cpu().numpy()
            vals = agents[i].predict_values(obs).detach().cpu() # type: ignore
            for j in range(vecs.shape[0]):
                data.append(dict(traj=id, depth=j, agent=i, encoding=vecs[j], val=vals[j].item()))
    # save encodings
    df = pd.DataFrame(data)
    print("saving encodings...")
    df.to_pickle(f"{trajs_dir}_enc.pkl")


if __name__ == "__main__":
    # Reg F
    convert_trajs("trajs/gen9vgc2024regf", [
        "results/saves_sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_a/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_a_to_f/64_teams/seed1/10027008.zip",
        "results/saves_sp/reg_c/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_c_to_f/64_teams/seed1/10027008.zip"
    ])

    # Reg I
    convert_trajs("trajs/gen9vgc2025regi", [
        "results/saves_sp/reg_i/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_a/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_a_to_i/64_teams/seed1/10027008.zip",
        "results/saves_sp/reg_f/64_teams/seed1/5013504.zip",
        "results/saves_sp/reg_f_to_i/64_teams/seed1/10027008.zip"
    ])
