#!/usr/bin/env python

from __future__ import print_function
import drmaa
import os
import jax.random as jrnd
import jax.numpy as jnp
import argparse
import pickle 
import pandas as pd
from pathlib import Path

def get_jt(session):
    jt = session.createJobTemplate()
    jt.remoteCommand = os.path.join(os.getcwd(), 'job.sh')
    jt.joinFiles = False
    jt.nativeSpecification = '-l h_rt=07:00:00 -l rmem=8G -P gpu -l gpu=1'

    return jt

def make_args(rkey, its, Nvgs, zgran, amps, f_name):
    return [str(its), " ".join(Nvgs), " ".join(zgran), " ".join(amps), str(f_name), str(rkey)]
    
def sample_cons_zgran(c, rkey):
    out = [jrnd.uniform(rkey, (1,), maxval=0.6, minval=0.1)] * c
    for i in range(1, c):
        rkey, _ = jrnd.split(rkey)
        out[i] = jrnd.uniform(rkey, (1,), maxval=out[i-1], minval=0.1)
    return out
        
    
def make_args_list(C, runs, seed_runs):
    its = ['5000', '8000', '10000', '10000', '10000']
    Nvgs = ['15', '10', '6', '4', '3']
    amp = 5.0
    zr = (0.1, 0.5)
    base_seed = 100
    args_list = []
    for c in range(C):
        for rep in range(runs):
            for i in range(0,seed_runs):
                raw_zgran = sample_cons_zgran(c+1, jrnd.PRNGKey(10*(rep*c + rep + base_seed)))
                zgran = ["%.3f" % zi for zi in raw_zgran] 
                amps = [str(amp)] * (c + 1)
                nvgs = Nvgs[:c+1]
                f_name = "c" + str(c+1) + "r" + str(rep) + "sr" + str(i)
                args = make_args(base_seed + seed_runs*rep + i, its[c], nvgs, zgran, amps, "results/" + f_name)
                args_list.append(args)
    return args_list


def main(expr, mode, C, runs, seed_runs):
    """
    Create a DRMAA session then submit a job.
    Note, need file called myjob.sh in the current directory.
    """
    args_list = make_args_list(C, runs, seed_runs)
    
    if expr not in ["fx", "weather", "tanks"]:
        raise ValueError("Not an expriment!")

    base_dir = "expr/" + expr
    os.chdir(base_dir)
    for n in ["best_logs", "best_results", "logs", "results"]:
        p = Path(n)
        try:
            p.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        
    
    if mode == "run_scan":
        with drmaa.Session() as s:
            for args in args_list:
                print(args)
                jt = get_jt(s)
                jt.args = args
                f_name = args[-2].strip("results/")
                jt.outputPath = os.path.join(os.getcwd(), ':logs/' + f_name)
                jt.jobName = f_name
                job_id = s.runJob(jt)
                print('Job {} submitted'.format(job_id))
                print('Cleaning up')
                s.deleteJobTemplate(jt)

    elif mode == "run_best":
        rep_runs = 2 
        with open("best_args.pkl", "rb") as f:
            best_args = pickle.load(f)
        
        with drmaa.Session() as s:
            for i in range(rep_runs):
    #             change key
                best_args[-1] = str(i)
                best_args[-2] = "best_results/rep" + str(i)
                jt = get_jt(s)
                print(best_args)
                jt.args = best_args
                jt.outputPath = os.path.join(os.getcwd(), ':best_logs/rep' + str(i))
                jt.jobName = "rep" + str(i)
                job_id = s.runJob(jt)
                print('Job {} submitted'.format(job_id))

                print('Cleaning up')
                s.deleteJobTemplate(jt)
        
                
                
    elif mode == "view_scan":
        if expr == "weather":
            res_df = pd.DataFrame(columns=["name", "zgran", "test NMSE", "train NMSE", "test NLPD","train NLPD", 
                                            "Chi NMSE", "Cam NMSE", "Chi NLPD", "Cam NLPD"])
        else:
            res_df = pd.DataFrame(columns=["name", "zgran", "test NMSE", "train NMSE", "test NLPD","train NLPD"])
        for i, args in enumerate(args_list):
            its, nvgs, zgran, amps, f_name, rkey = args 
            try:
                with open(f_name + "res.pkl", "rb") as f:
                    res_dict = pickle.load(f)
                
                with open("logs/" + f_name.strip("results/"), "r") as f:
                    lines = f.readlines()
                
                if expr == "weather":
                    spec_met = []
#                     print(res_dict["Chimet NMSE"], res_dict["Cambermet NMSE"])
#                     for p in ["Chimet NMSE", "Cambermet NMSE", "Chimet NLPD", "Cambermet NLPD"]:
# #                         
#                         spec_met.append([float(str(s).strip("\n").strip(p + ": ")) for s in lines if s.startswith(p)][0])

                    if any(line.startswith("nan") for line in lines):
                        res_df.loc[i] = [f_name.strip("results/"), zgran] + [jnp.nan] * 8
                    else:
                        res_df.loc[i] = [f_name.strip("results/"), zgran, res_dict["test NMSE"],
                            res_dict["train NMSE"], res_dict["test NLPD"], res_dict["train NLPD"],
                              res_dict["Cambermet NMSE"], res_dict["Cambermet NLPD"], res_dict["Chimet NMSE"],
                                res_dict["Chimet NLPD"]] + spec_met

                else:
                    res_df.loc[i] = [f_name.strip("results/"), zgran, res_dict["test NMSE"],
                        res_dict["train NMSE"], res_dict["test NLPD"], res_dict["train NLPD"], ]

            except FileNotFoundError:
                print(f_name + " doesn't exist, probably killed!")
                    
        res_df = res_df.sort_values("train NMSE")
        print(res_df)
        res_df.to_csv("scan_results.txt")
        with open("best_args.pkl", "wb") as f:
            pickle.dump(args_list[int(res_df.iloc[0].name)], f)
        
        if expr == "weather":
            res_df = res_df.dropna()
        
            sum_df = pd.DataFrame(columns=["name", "N", "test NMSE", "train NMSE", "test NLPD","train NLPD", 
                                                "Cam NMSE",  "Cam NLPD", "Chi NMSE", "Chi NLPD"])
            sum_df_std = pd.DataFrame(columns=["name", "N", "test NMSE", "train NMSE", "test NLPD","train NLPD", 
                                                "Cam NMSE",  "Cam NLPD", "Chi NMSE", "Chi NLPD"])
            for c in range(C):
                for r in range(runs):
                    sub_df = res_df[res_df['name'].str.contains(f"c{c+1}r{r}")]
                    if len(sub_df) < 3: continue
                    sub_df = sub_df.iloc[:3]
                    sum_df.loc[c*C + r] = [f"c{c+1}r{r}", len(sub_df)] + list(sub_df.mean())
                    sum_df_std.loc[c*C + r] = [f"c{c+1}r{r}", len(sub_df)] + list(sub_df.std())
        
            sort_idx = sum_df["train NLPD"].argsort()
            sum_df = sum_df.iloc[sort_idx]
            sum_df_std = sum_df_std.iloc[sort_idx]
            print(sum_df)
            print(sum_df_std)

            best_df = res_df[res_df['name'].str.contains(sum_df.iloc[0]["name"])]
            print(args_list[int(best_df.sort_values("train NLPD").iloc[0].name)])
            
            
    elif mode == "view_best":
        rep_names = os.listdir("best_logs")
        res_df = pd.DataFrame(columns=["name", "test NMSE", "train NMSE", "test NLPD","train NLPD"])
        for i, name in enumerate(rep_names):
            try:
                with open("best_results/" + name + "res.pkl", "rb") as f:
                    res_dict = pickle.load(f)
                res_df.loc[i] = [name, res_dict["test NMSE"],
                                res_dict["train NMSE"], res_dict["test NLPD"], res_dict["train NLPD"]]
            except FileNotFoundError:
                print(name + " doesn't exist, probably killed!")

        
        res_df = res_df.sort_values("train NLPD")
        print(res_df)
        res_df.to_csv("best_scan_results.txt")
        
    else:
        raise ValueError("Not a mode")        
                
                
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Job controller")
    parser.add_argument("--mode", default="res", type=str)
    parser.add_argument("--expr", default="weather", type=str)
    parser.add_argument("--C", default=4, type=int)
    parser.add_argument("--runs", default=10, type=int)
    parser.add_argument("--seed_runs", default=3, type=int)
    args = parser.parse_args()
    main(args.expr, args.mode, args.C, args.runs, args.seed_runs)
