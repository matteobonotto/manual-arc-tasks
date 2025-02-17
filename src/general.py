import random
import string
from datetime import datetime
from typing import List, Tuple, Dict, Callable, Any
from time import time
import os
import pickle
from copy import deepcopy
import json
from tqdm import tqdm
from pathlib import Path
import shutil
from operator import itemgetter
from itertools import chain

from .types import Grid, JSONTask, GridPairs

from typing import Dict, List, Tuple
from .types import Grid, GridPairs, JSONTask

# from .general import random_ARC_name


def adjust_data_format(
    tasks: Dict[str, JSONTask] | List[JSONTask],
) -> Tuple[Dict[str, GridPairs], Dict[str, GridPairs]]:
    """Load all data from a given folder in a dictionary with keys the tasks.
    (Data is small (I guess 2*10**6 at most) so it's okay to keep it in memory at runtime.)
    (Must create a Kaggle compatible one at some point)
    """
    # solve the path
    # root_directory = os.getcwd()  # you must run this from the project root directory
    # data_folder_path = os.path.join(
    #     root_directory,
    #     data_folder,
    #     data_sub_folder,
    # )
    # print(data_folder_path)
    # # get all data files names in the folder
    # file_names = os.listdir(data_folder_path)
    # create a dictionary with the examples for each task and another one with the tests
    examples_data: Dict = {}
    truth_data: Dict = {}
    if isinstance(tasks, dict):
        task_ids = list(tasks.keys())
        tasks = list(tasks.values())
    else:
        task_ids = [random_ARC_name(15) for _ in range(len(tasks))]
    count = 0
    for i, data in enumerate(tasks):
        count += 1
        task_name = task_ids[i]
        # add the training examples
        new_grid_pairs = []
        for pair in data["train"]:
            new_grid_pairs.append((pair["input"], pair["output"]))
        examples_data[task_name] = new_grid_pairs
        # add the test examples
        new_grid_pairs = []
        if "output" in data["test"][0].keys():
            for pair in data["test"]:
                new_grid_pairs.append((pair["input"], pair["output"]))
        truth_data[task_name] = new_grid_pairs
    return examples_data, truth_data


def task_max_grid_shape(task: JSONTask) -> Tuple[int, int]:
    """Compute the maximum shape around all the grids of a given example"""
    shape_col = []
    shape_row = []
    for k in task.keys():
        for i in range(len(task[k])):
            for kk in ["input", "output"]:
                shape_row.append(len(task[k][i][kk]))
                shape_col.append(len(task[k][i][kk][0]))
    return max(shape_row), max(shape_col)


def get_colors(grid: Grid) -> List[int]:
    """get the colors of a grid"""
    return list(set(chain.from_iterable(grid)))


def check_io_equal_grid_size(task: JSONTask) -> bool:
    grid_pairs = task["train"] + task["test"]
    return any(
        [grid_shapes(x["input"]) == grid_shapes(x["output"]) for x in grid_pairs]
    )


def filter_dict_by_idx(d: Dict[Any, Any], idx: List[int] | int) -> Dict[Any, Any]:
    if not isinstance(idx, int):
        if len(idx) == 1:
            return {k: v for i, (k, v) in enumerate(d.items()) if i in idx}
        else:
            return dict(itemgetter(*idx)(list(d.items())))
    else:
        return {k: v for i, (k, v) in enumerate(d.items()) if i == idx}


def timer_func(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator to time function execution

    :param func: function to be timed
    :return: timed function

    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f"{func.__name__}() executed in {(t2-t1):.6f}s")
        print("")
        return result

    return wrapper


def grid_shapes(grid: Grid):
    return (len(grid), len(grid[0]))


def clear_folder(path):
    for f in Path(path).glob("*"):
        if f.is_file():
            f.unlink()
        else:
            shutil.rmtree(f)


def touch_dir(path):
    Path(path).mkdir(exist_ok=True, parents=True)


def random_ARC_name(len_str: int = 10) -> str:
    return "".join(
        random.choice(string.ascii_lowercase + string.digits) for _ in range(len_str)
    )


def random_name():
    return str(abs(hash(datetime.now())))


def task_to_grid_pairs(task: JSONTask) -> GridPairs:
    grid_pairs_in = [(x["input"], x["output"]) for x in task["train"]]
    if "output" in task["test"][0].keys():
        grid_pairs_out = [(x["input"], x["output"]) for x in task["test"]]
    else:
        grid_pairs_out = [(x["input"], []) for x in task["train"]]
    grid_pairs: List[Tuple[Grid, Grid]] = grid_pairs_in + grid_pairs_out
    return grid_pairs


def convert_grids_to_task(
    grids_in: List[Grid],
    grids_out: List[Grid],
    no_tests: int = 1,
) -> JSONTask:
    output: Dict[str, List[Dict[str, Grid]]] = {"train": [], "test": []}
    output["train"] = [
        {"input": grid_in, "output": grid_out}
        for grid_in, grid_out in zip(grids_in[:-no_tests], grids_out[:-no_tests])
    ]
    output["test"] = [
        {"input": grid_in, "output": grid_out}
        for grid_in, grid_out in zip(grids_in[-no_tests:], grids_out[-no_tests:])
    ]
    return output


def generate_unique_id(dummy=0):
    return str(abs(hash(datetime.now())))


def load_pickled_tasks(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, dict):
        return list(data.values()), list(data.keys())
    else:
        return data, list(map(generate_unique_id, range(len(data))))


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def load_task(path):
    task = load_json(path)
    # some tasks have the "name" among the keys in addition to "train", "test"
    # like {'train': [{...}, {...}, {...}], 'test': [{...}], 'name': 'c35c1b4c'}
    return {k: task[k] for k in ["train", "test"]}, str(path).split("/")[-1].split(".")[
        0
    ]


def load_existing_tasks(path, VERBOSE=True):
    tasks, tasks_id = [], []
    # Traverse the directory structure
    for root, dirs, files in os.walk(path):
        if VERBOSE:
            pbar = tqdm(files, total=len(files), desc="Loading data")
        else:
            pbar = files
        for file in pbar:
            # Check if the file is a pickle file (usually ends with .pkl or .pickle)
            if file.endswith(".pkl") or file.endswith(".json"):
                file_path = os.path.join(root, file)

                # Read the pickle file and store it in the dictionary
                if file.endswith(".pkl"):
                    tasks_, ids_ = load_pickled_tasks(file_path)
                else:
                    tasks_, ids_ = load_task(file_path)
                (
                    tasks.extend(tasks_)
                    if isinstance(tasks_, list)
                    else tasks.append(tasks_)
                )
                (
                    tasks_id.extend(ids_)
                    if isinstance(tasks_, list)
                    else tasks_id.append(ids_)
                )
    return (tasks, tasks_id)
