import json
import os
import random
import shutil
import string
import subprocess
import sys
import urllib.request
from datetime import date, datetime, timedelta, timezone
from enum import Enum
from glob import glob
from logging import Logger
from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any, Optional

import cpuinfo
import numpy as np
import torch
import yaml
from attrdict import AttrDict
from colorama import Fore, Style
from dateutil.parser import parse as parse_date
from IPython import get_ipython
from pyunpack import Archive
from torchinfo import summary

from graph_tools.utils.logger import get_logger, kill_logger


def now() -> datetime:
    JST = timezone(timedelta(hours=9))
    return datetime.now(JST)


def is_notebook() -> bool:
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMInteractiveShell":
            return True  # Jupyter notebook qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal ipython
        elif "google.colab" in sys.modules:
            return True  # Google Colab
        else:
            return False
    except NameError:
        return False


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


def get_enum_from_value(enum_class, value: Any):
    value_map = {item.value: item for item in enum_class}
    return value_map[value]


class Phase(Enum):
    DEV = "dev"
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"
    SUBMISSION = "submission"


class TimeUnit(Enum):
    ONE_SECOND = "s1"
    ONE_MINUTE = "m1"
    FIVE_MINUTES = "m5"
    TEN_MINUTES = "m10"
    QUARTER_HOUR = "m15"
    HALF_HOUR = "m30"
    ONE_HOUR = "m60"
    ONE_DAY = "d1"
    MILLISECONDS = "milliseconds"
    NONE = "none"


class Config(object):
    NVIDIA_SMI_DEFAULT_ATTRIBUTES = (
        "index",
        "uuid",
        "name",
        "timestamp",
        "memory.total",
        "memory.free",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
    )

    def __init__(self, config_path: PathLike, ex_args: dict = {}, silent=False):
        self.__load_config__(config_path, ex_args, silent)

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__config__:
            if isinstance(self.__config__[__name], dict):
                return AttrDict(self.__config__[__name])
            else:
                return self.__config__[__name]
        else:
            raise AttributeError(f"'Config' object has no attribute '{__name}'")

    @classmethod
    def get_hash(cls, size: int = 12) -> str:
        chars = string.ascii_lowercase + string.digits
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    @classmethod
    def now(cls) -> datetime:
        JST = timezone(timedelta(hours=9))
        return datetime.now(JST)

    def __get_device__(self):
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def __load_config__(self, config_path: PathLike, ex_args: dict = {}, silent=False):
        self.__config__ = AttrDict(yaml.safe_load(open(config_path)))
        if len(ex_args) > 0:
            self.__config__ = self.__config__ + ex_args
        self.__config__["config_path"] = Path(config_path)
        self.__config__["timestamp"] = self.now()
        self.__config__["train"]["device"] = torch.device(self.__get_device__())
        self.__config__["log"]["log_dir"] = (
            Path(self.__config__["log"]["log_dir"])
            / f"{self.__config__['train']['exp_name']}_{self.__config__['timestamp'].strftime('%Y%m%d%H%M%S')}"
        )
        self.__config__["log"]["log_file"] = (
            Path(self.__config__["log"]["log_dir"]) / self.__config__["log"]["log_filename"]
        )
        self.__config__["weights"]["global_weights_dir"] = Path(self.__config__["weights"]["global_weights_dir"])
        self.__config__["weights"]["log_weights_dir"] = Path(self.__config__["log"]["log_dir"]) / "weights"
        self.__config__["data"]["data_path"] = Path(self.__config__["data"]["data_path"])
        self.__config__["data"]["cache_path"] = Path(self.__config__["data"]["cache_path"])
        self.__config__["backup"]["backup_dir"] = Path(self.__config__["backup"]["backup_dir"])
        self.__config__["output"]["out_dir"] = Path(self.__config__["output"]["out_dir"])
        self.__config__["log"]["loggers"] = {}

        if hasattr(self, "__logger") and isinstance(self.__logger, Logger):
            kill_logger(self.__logger)
        self.__config__["log"]["loggers"]["logger"] = get_logger(
            name="config", logfile=self.__config__["log"]["log_file"], silent=silent
        )
        self.__config__["log"]["logger"] = self.__config__["log"]["loggers"]["logger"]

        # mkdir
        self.__config__["data"]["data_path"].mkdir(parents=True, exist_ok=True)
        self.__config__["data"]["cache_path"].mkdir(parents=True, exist_ok=True)
        self.__config__["backup"]["backup_dir"].mkdir(parents=True, exist_ok=True)
        self.__config__["output"]["out_dir"].mkdir(parents=True, exist_ok=True)
        self.__config__["weights"]["global_weights_dir"].mkdir(parents=True, exist_ok=True)
        self.__config__["weights"]["log_weights_dir"].mkdir(parents=True, exist_ok=True)

        self.log.logger.info("====== show config =========")
        attrdict_attrs = list(dir(AttrDict()))
        for key, value in self.__config__.items():
            if key not in attrdict_attrs:
                if isinstance(value, dict):
                    for key_2, value_2 in value.items():
                        if key_2 not in attrdict_attrs:
                            self.log.logger.info(f"config: {key:15s}-{key_2:20s}: {value_2}")
                else:
                    self.log.logger.info(f"config: {key:35s}: {value}")
        self.log.logger.info("============================")

        # CPU info
        self.describe_cpu()

        # GPU info
        if torch.cuda.is_available():
            self.describe_cuda()

        # Mac M1 Sillicon
        if torch.backends.mps.is_available():
            self.describe_m1_silicon()

        # fix seed
        self.fix_seed(self.train.seed)

    @property
    def config_dict(self):
        return self.__config__

    def describe_cpu(self):
        self.log.logger.info("====== cpu info ============")
        for key, value in cpuinfo.get_cpu_info().items():
            self.log.logger.info(f"CPU INFO: {key:20s}: {value}")
        self.log.logger.info("============================")

    def describe_cuda(self, nvidia_smi_path="nvidia-smi", no_units=True):
        try:
            keys = self.NVIDIA_SMI_DEFAULT_ATTRIBUTES
            nu_opt = "" if not no_units else ",nounits"
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True)
            lines = output.decode().split("\n")
            lines = [line.strip() for line in lines if line.strip() != ""]
            lines = [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]

            self.log.logger.info("====== show GPU information =========")
            for line in lines:
                for k, v in line.items():
                    self.log.logger.info(f"{k:25s}: {v}")
            self.log.logger.info("=====================================")
        except CalledProcessError:
            self.log.logger.info("====== show GPU information =========")
            self.log.logger.info("  No NVIDIA GPU was found.")
            self.log.logger.info("=====================================")

    def describe_m1_silicon(self):
        self.log.logger.info("====== show GPU information =========")
        self.log.logger.info("  Mac-M1 GPU is available.")
        self.log.logger.info("=====================================")

    def describe_model(self, model: torch.nn.Module, input_size: tuple = (), input_data=None):
        if input_data is None:
            summary_str = summary(
                model,
                input_size=input_size,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )
        else:
            summary_str = summary(
                model,
                input_data=input_data,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )

        for line in summary_str.__str__().split("\n"):
            self.log.logger.info(line)

    def backup_logs(self):
        """copy log directory to config.backup"""
        backup_dir = Path(self.backup.backup_dir)
        if backup_dir.exists():
            shutil.rmtree(str(backup_dir))
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.log.log_dir, self.backup.backup_dir)

    def add_logger(self, name: str, silent: bool = False):
        self.__config__["log"]["loggers"][name] = get_logger(
            name=name, logfile=self.__config__["log"]["log_file"], silent=silent
        )
        self.__config__["log"][name] = self.__config__["log"]["loggers"][name]

    def fix_seed(self, seed=42):
        self.log.logger.info(f"seed - {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True


def timedelta2HMS(total_sec: int) -> str:
    h = total_sec // 3600
    m = total_sec % 3600 // 60
    s = total_sec % 60
    return f"{h:2d}h {m:2d}m {s:2d}s"


def __show_progress__(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = "=" * bar_num
    if bar_num != max_bar:
        progress_element += ">"
    bar_fill = " "
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        Fore.LIGHTCYAN_EX,
        f"[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r",
        end="",
    )


def download(url: str, filepath: str):
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    print(Fore.LIGHTGREEN_EX, "download from:", end="")
    print(Fore.WHITE, url)
    urllib.request.urlretrieve(url, filepath, __show_progress__)
    print("")  # 改行
    print(Style.RESET_ALL, end="")


def un7zip(src_path: str, dst_path: str):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    Archive(src_path).extractall(dst_path)
    for dirname, _, filenames in os.walk(dst_path):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def isint(s: str) -> bool:
    """Check the argument string is integer or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is integer or not.
    """
    try:
        int(s, 10)
    except ValueError:
        return False
    else:
        return True


def isfloat(s: str) -> bool:
    """Check the argument string is float or not.

    Args:
        s (str): string value.

    Returns:
        bool: If the given string is float or not.
    """
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


def datetime2str(dt_data: Optional[datetime], time_unit: TimeUnit) -> str:
    """convert datetime into str

    Args:
        dt_data (datetime)
        time_unit (TimeUnit)

    Return:
        datetime string
    """
    if dt_data is not None:
        if time_unit == TimeUnit.ONE_DAY:
            return dt_data.strftime("%Y-%m-%d")
        elif time_unit == TimeUnit.ONE_HOUR:
            return dt_data.strftime("%Y-%m-%d %H:00:00")
        elif time_unit == TimeUnit.ONE_MINUTE:
            return dt_data.strftime("%Y-%m-%d %H:%M:00")
        elif time_unit == TimeUnit.FIVE_MINUTES:
            year, month, day, hour, minute = dt_data.year, dt_data.month, dt_data.day, dt_data.hour, dt_data.minute
            minute = int(dt_data.minute // 5 * 5)
            date = datetime(year, month, day, hour, minute, 0)
            return date.strftime("%Y-%m-%d %H:%M:00")
        elif time_unit == TimeUnit.TEN_MINUTES:
            year, month, day, hour, minute = dt_data.year, dt_data.month, dt_data.day, dt_data.hour, dt_data.minute
            minute = int(dt_data.minute // 10 * 10)
            date = datetime(year, month, day, hour, minute, 0)
            return date.strftime("%Y-%m-%d %H:%M:00")
        elif time_unit == TimeUnit.QUARTER_HOUR:
            year, month, day, hour, minute = dt_data.year, dt_data.month, dt_data.day, dt_data.hour, dt_data.minute
            minute = int(dt_data.minute // 15 * 15)
            date = datetime(year, month, day, hour, minute, 0)
            return date.strftime("%Y-%m-%d %H:%M:00")
        elif time_unit == TimeUnit.HALF_HOUR:
            year, month, day, hour, minute = dt_data.year, dt_data.month, dt_data.day, dt_data.hour, dt_data.minute
            minute = int(dt_data.minute // 30 * 30)
            date = datetime(year, month, day, hour, minute, 0)
            return date.strftime("%Y-%m-%d %H:%M:00")
        elif time_unit == TimeUnit.ONE_SECOND:
            return dt_data.strftime("%Y-%m-%d %H:%M:%S")
        elif time_unit == TimeUnit.MILLISECONDS:
            return dt_data.strftime("%Y-%m-%d %H:%M:%S.%f")
        elif time_unit == TimeUnit.NONE:
            return "NONE"
        else:
            raise RuntimeError(f"Unknown TimeUnit: {time_unit}")
    else:
        return ""


def str2datetime(ts_str: str) -> Optional[datetime]:
    """convert timestamp string into datetime"""
    try:
        return parse_date(ts_str)
    except Exception:
        return None


class JsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, "__iter__"):
            return list(obj)
        elif isinstance(obj, datetime):
            return obj.strftime("%Y%m%d %H:%M:%S.%f")
        elif isinstance(obj, date):
            return datetime(obj.year, obj.month, obj.day, 0, 0, 0).strftime("%Y%m%d %H:%M:%S.%f")
        else:
            return super().default(obj)
