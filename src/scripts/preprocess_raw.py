import os
import sys
from pathlib import Path
import json
import random
from string import ascii_lowercase
import shutil
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.visualization import ArcPlotter, MatplotlibARCPlot

# PWD = "gen_tasks/raw"
DIGITS = [str(x) for x in range(10)]
LETTERS = list(ascii_lowercase)

def arc_name(n:int=10) -> str:
    return ''.join([random.choice(DIGITS+LETTERS) for _ in range(n)])


def load_tasks(path):
    tasks = {}
    for file in Path(path).iterdir():
        if "json" in file.suffix:
            tasks.update(
                {arc_name(20):json.load(open(file, "r"))}
            )
    return tasks

if __name__ == "__main__":
    data_dir = "gen_tasks"
    plotter = MatplotlibARCPlot()

    # read tasks
    tasks = load_tasks(f"{data_dir}/raw")

    # filter duplicates
    unique_tasks = {f"mb_{arc_name(13)}":json.loads(x.replace("'", '"')) for x in set([str(y) for y in tasks.values()])}

    #remove all the tasks and store the unique ones
    Path(f"{data_dir}/preprocessed").mkdir(parents=True, exist_ok=True)
    for id, task in tqdm(unique_tasks.items(), total=len(unique_tasks)):
        json.dump(task, open(f"{data_dir}/preprocessed/{id}.json", 'w'))
        plotter.show(
            task=task, title=id, taskname=id, savefig=True, dirsave=f"{data_dir}/preprocessed/"
        )


    for file in Path(f"{data_dir}/raw").iterdir():
        try:
            os.unlink(file)
        except:
            pass
