import os
import sys
from pathlib import Path
import json
import random
from string import ascii_lowercase
import shutil
from tqdm import tqdm
from typing import TypeAlias, List, Dict, Set

sys.path.append(os.getcwd())
from src.visualization import ArcPlotter, MatplotlibARCPlot

JSONTask: TypeAlias = Dict[str, List[Dict[str, List[List[int]]]]]

# PWD = "gen_tasks/raw"
DIGITS = [str(x) for x in range(10)]
LETTERS = list(ascii_lowercase)


def arc_name(n: int = 10) -> str:
    return "".join([random.choice(DIGITS + LETTERS) for _ in range(n)])


def load_tasks(path):
    tasks = {}
    for file in Path(path).iterdir():
        if "json" in file.suffix:
            tasks.update({arc_name(20): json.load(open(file, "r"))})
    return tasks

def check_existance(task: JSONTask, already_present: Set[str]) -> bool:
    if str(next(iter(task.values()))) not in already_present:
        return True
    else:
        return False


if __name__ == "__main__":
    data_dir = "gen_tasks"
    plotter = MatplotlibARCPlot()

    # read tasks
    tasks = load_tasks(f"{data_dir}/raw")

    # filter duplicates
    unique_tasks = {
        f"mb_{arc_name(13)}": json.loads(x.replace("'", '"'))
        for x in set([str(y) for y in tasks.values()])
    }

    # remove all the tasks and store the unique ones
    Path(f"{data_dir}/preprocessed").mkdir(parents=True, exist_ok=True)
    already_present = set([str(y) for y in load_tasks(f"{data_dir}/preprocessed").values()])
    for id, task in tqdm(unique_tasks.items(), total=len(unique_tasks)):
        if check_existance(task, already_present):
            json.dump(task, open(f"{data_dir}/preprocessed/{id}.json", "w"))
            plotter.show(
                task=task,
                title=id,
                taskname=id,
                savefig=True,
                dirsave=f"{data_dir}/preprocessed/",
            )

    for file in Path(f"{data_dir}/raw").iterdir():
        try:
            os.unlink(file)
        except:
            pass

    # push to hf hub
