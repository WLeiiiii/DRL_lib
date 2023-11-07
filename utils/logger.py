import csv
import os
import shutil
from collections import defaultdict

import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter


def prepare_file(prefix, suffix):
    file_name = f"{prefix}.{suffix}"
    if os.path.exists(file_name):
        os.remove(file_name)
    return file_name


def format_type(key, value, ty):
    if ty == "int":
        value = int(value)
        return f"{key}: {value}"
    elif ty == "float":
        return f"{key}: {value:.04f}"
    elif ty == "time":
        return f"{key}: {value:04.1f} s"
    else:
        return f"Invalid format type: {ty}"


COMMON_TRAIN_FORMAT = [
    ("episode", "E", "int"),
    ("episode_steps", "ES", "int"),
    ("step", "S", "int"),
    ("episode_reward", "R", "float"),
    ("duration", "D", "time"),
]

COMMON_EVAL_FORMAT = [
    ("episode", "E", "int"),
    ("step", "S", "int"),
    ("mean_reward", "R", "float"),
    ("mean_step", "ES", "float"),
]

AGENT_TRAIN_FORMAT = {
    "dqn": [
        ("batch_reward", "BR", "float"),
        ("batch_cost", "BC", "float"),
        ("batch_r-c", "BR-BC", "float"),
        ("cost_rate", "CRATE", "float"),
        ("loss", "LOSS", "float"),
    ],
    "double_dqn": [
        ("batch_reward", "BR", "float"),
        ("batch_cost", "BC", "float"),
        ("batch_r-c", "BR-BC", "float"),
        ("cost_rate", "CRATE", "float"),
        ("loss", "LOSS", "float"),
    ],
}


class Logger:
    def __init__(self, log_dir, time, save_tb=False, log_frequency=10000, agent="dqn"):
        self._log_dir = log_dir
        self._log_frequency = log_frequency
        if save_tb:
            tb_dir = os.path.join(log_dir, "data/" + agent + "/" + time + "/" + "tb")
            if os.path.exists(tb_dir):
                try:
                    shutil.rmtree(tb_dir)
                except Warning:
                    print("logger.py warning: Unable to remove tb directory")
                    pass

            self._sw = SummaryWriter(tb_dir)
        else:
            self._sw = None

        assert agent in AGENT_TRAIN_FORMAT
        train_format = COMMON_TRAIN_FORMAT + AGENT_TRAIN_FORMAT[agent]
        self._train_mg = MetersGroup(os.path.join(tb_dir, "train"), formating=train_format)
        self._eval_mg = MetersGroup(os.path.join(tb_dir, "eval"), formating=COMMON_EVAL_FORMAT)
        pass

    def _should_log(self, step, log_frequency):
        log_frequency = log_frequency or self._log_frequency
        return step % log_frequency == 0

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step, n=1, log_frequency=1):
        if not self._should_log(step, log_frequency):
            return
        assert key.startswith("train") or key.startswith("eval")
        if type(value) == torch.Tensor:
            value = value.item()
        self._try_sw_log(key, value / n, step)
        mg = self._train_mg if key.startswith("train") else self._eval_mg
        mg.log(key, value, n)
        pass

    def dump(self, step, save=True, ty=None):
        if ty is None:
            self._train_mg.dump(step, "train", save)
            self._eval_mg.dump(step, "eval", save)
        elif ty == "eval":
            self._eval_mg.dump(step, "eval", save)
        elif ty == "train":
            self._train_mg.dump(step, "train", save)
        else:
            raise f"Invalid log type: {ty}"
        pass


class MetersGroup:
    def __init__(self, file_name, formating):
        self._csv_file_name = prepare_file(file_name, "csv")
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = open(self._csv_file_name, "w")
        self._csv_writer = None
        pass

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)
        pass

    def dump(self, step, prefix, save=True):
        if len(self._meters) == 0:
            return
        if save:
            data = self._prime_meters()
            data["step"] = step
            self._dump_to_csv(data)
            self._dump_to_console(data, prefix)
        self._meters.clear()

    def _prime_meters(self):
        data = dict()
        for key, meter in self._meters.items():
            if key.startswith("train"):
                key = key[len("train") + 1:]
            else:
                key = key[len("eval") + 1:]

            key = key.replace("/", "_")
            data[key] = meter.value()
        return data

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=sorted(data.keys()), restval=0.0)
            self._csv_writer.writeheader()
        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _dump_to_console(self, data, prefix):
        prefix = colored(prefix, "yellow" if prefix == "train" else "green")
        pieces = [f"| {prefix: <14}"]
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(format_type(disp_key, value, ty))
        print(" | ".join(pieces))
        pass


class AverageMeter:
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)
