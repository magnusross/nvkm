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
    jt.joinFiles = True
    jt.nativeSpecification = '-l h_rt=07:00:00 -l rmem=8G -P gpu -l gpu=1'

    return jt

def make_args(rkey, its, Nvgs, zgran, amps, f_name):
    return [str(its), " ".join(Nvgs), " ".join(zgran), " ".join(amps), str(f_name), str(rkey)]
    
def make_args_list(C, runs):
    its = ['5000', '8000', '10000', '10000']
    Nvgs = ['15', '10', '6', '4', '3']
    amp = 5.0
    zr = (0.1, 0.5)
    args_list = []
    for c in range(C):
        for rep in range(runs):
            raw_zgran = jrnd.uniform(jrnd.PRNGKey(rep*10), (c+1, 1), minval=zr[0], maxval=zr[1])
            zgran = ["%.3f" % zi for zi in raw_zgran] 
            amps = [str(amp)] * (c + 1)
            nvgs = Nvgs[:c+1]
            f_name = "c" + str(c+1) + "r" + str(rep)
            args = make_args(rep, its[c], nvgs, zgran, amps, "results/" + f_name)
            args_list.append(args)
    return args_list


def main(expr, mode, C, runs):
    """
    Create a DRMAA session then submit a job.
    Note, need file called myjob.sh in the current directory.
    """
    args_list = make_args_list(C, runs)
    
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
        res_df = pd.DataFrame(columns=["name", "zgran", "test NMSE", "train NMSE", "test NLPD","train NLPD"])
        for i, args in enumerate(args_list):
            its, nvgs, zgran, amps, f_name, rkey = args 
            try:
                with open(f_name + "res.pkl", "rb") as f:
                    res_dict = pickle.load(f)
                
                with open("logs/" + f_name.strip("results/"), "rb") as f:
                    lines = f.readlines()
                
                if any(line.startswith(b"nan") for line in lines):
                    res_df.loc[i] = [f_name.strip("results/"), zgran, jnp.nan, jnp.nan, jnp.nan, jnp.nan]
                else:
                    res_df.loc[i] = [f_name.strip("results/"), zgran, res_dict["test NMSE"],
                        res_dict["train NMSE"], res_dict["test NLPD"], res_dict["train NLPD"]]
                    
                
            except FileNotFoundError:
                print(f_name + " doesn't exist, probably killed!")
                    
        res_df = res_df.sort_values("train NMSE")
        print(res_df)
        res_df.to_csv("scan_results.txt")
        
        with open("best_args.pkl", "wb") as f:
            pickle.dump(args_list[int(res_df.iloc[0].name)], f)
            
    elif mode == "view_best":
        raise NotImplementedError
        
    else:
        raise ValueError("Not a mode")        
                
                
                


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Job controller")
    parser.add_argument("--mode", default="res", type=str)
    parser.add_argument("--expr", default="weather", type=str)
    parser.add_argument("--C", default=4, type=int)
    parser.add_argument("--runs", default=10, type=int)
    args = parser.parse_args()
    main(args.expr, args.mode, args.C, args.runs)