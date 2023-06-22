from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from os import PathLike
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple, Union

import dgl
import networkx as nx
import numpy as np
import pandas as pd
import torch
from dateutil.parser import parse as parse_date
from scipy.special import softmax
from torch.utils.data import Dataset

from graph_tools.utils.utils import Config, Phase, TimeUnit, datetime2str, is_notebook, str2datetime

if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm

Timestamp = str


class Label(Enum):
    BENIGN = "BENIGN"
    ANOMALY = "ANOMALY"
    MIXED = "MIXED"
    UNDEFINED = "UNDEFINED"

    @staticmethod
    def to_label(value: Any) -> Label:
        if isinstance(value, Label):
            return value
        elif isinstance(value, int):
            if value == 0:
                return Label.BENIGN
            elif value == 1:
                return Label.ANOMALY
            else:
                raise ValueError(f"unexpected value: {value}")
        elif isinstance(value, str):
            if value.lower() == "benign":
                return Label.BENIGN
            elif value.lower() == "normal":
                return Label.BENIGN
            elif value.lower() == "0":
                return Label.BENIGN
            elif value.lower() == "anomaly":
                return Label.ANOMALY
            elif value.lower() == "anomalous":
                return Label.ANOMALY
            elif value.lower() == "malware":
                return Label.ANOMALY
            elif value.lower() == "1":
                return Label.ANOMALY
            elif value.lower() == "undefined":
                return Label.UNDEFINED
            else:
                raise ValueError(f"unexpected value: {value}")
        else:
            raise ValueError(f"unexpected value: {value}")


@dataclass
class TimeWindow(object):
    start: datetime
    end: datetime
    delta: timedelta

    def __hash__(self) -> int:
        return self.start.__hash__() + self.end.__hash__() + self.delta.__hash__()

    def __str__(self) -> str:
        s = datetime2str(self.start, TimeUnit.MILLISECONDS)
        e = datetime2str(self.end, TimeUnit.MILLISECONDS)
        return f"<TimeWindow: {s} -> {e} ({str(self.delta)})"

    def __repr__(self) -> str:
        return self.__str__()

    def size(self) -> timedelta:
        return self.end - self.start

    def isin(self, t: datetime):
        return self.start <= t < self.end

    def next_window(self) -> TimeWindow:
        """get next time window

        Returns:
            TimeWindow: next time window
        """
        start = self.start + self.delta
        end = self.end + self.delta
        return TimeWindow(start, end, self.delta)

    def copy(self) -> TimeWindow:
        """create copy of the instance

        Returns:
            TimeWindow: new instance
        """
        return TimeWindow(self.start, self.end, self.delta)

    def to_dict(self) -> Dict[str, Any]:
        """dump parameters into dict

        Returns:
            Dict[str, Any]: parameters as dict
        """
        return {
            "start": datetime2str(self.start, TimeUnit.MILLISECONDS),
            "end": datetime2str(self.end, TimeUnit.MILLISECONDS),
            "delta": self.delta.total_seconds(),
        }

    @classmethod
    def load_dict(cls, dict_data: Dict[str, Any]) -> TimeWindow:
        """load dict and return a new TimeWindow instance

        Args:
            dict_data (Dict[str, Any]): dict data dumped by TimeWindow object

        Returns:
            TimeWindow: a new object of TimeWindow
        """
        return cls(
            start=parse_date(dict_data["start"]),
            end=parse_date(dict_data["end"]),
            delta=timedelta(seconds=dict_data["delta"]),
        )


class BaseDataset(ABC, Dataset):
    def __init__(self, config: Config, phase=Phase.TRAIN):
        super().__init__()
        self.config = config
        self.config.add_logger("dataset_log")
        self.dataset_path = Path(self.config.data.data_path)
        self.valid_size = self.config.train.valid_size
        self.test_size = self.config.train.test_size
        self.phase: Phase = phase

        self.train_data: Dict = {}
        self.valid_data: Dict = {}
        self.test_data: Dict = {}
        self.dev_data: Dict = {}
        # self.train_data, self.valid_data, self.test_data =
        #   self.__load_data__(self.dataset_path, config.train.test_size, config.train.valid_size)
        # self.dev_data = {idx: self.train_data[idx] for idx in range(1000)}

    def __load_data__(self, dataset_path: PathLike, test_size: float, valid_size: float):
        """load train and test data
        Returns:
            train_data, valid_data, test_data: dict[index, namedtuple[data, label]]
        """
        # return train_data, test_data, label_data
        raise NotImplementedError()

    def __len__(self) -> int:
        if self.phase == Phase.TRAIN:
            return len(self.train_data)
        elif self.phase == Phase.VALID:
            return len(self.valid_data)
        elif self.phase == Phase.TEST:
            return len(self.test_data)
        elif self.phase == Phase.DEV:
            return len(self.dev_data)
        elif self.phase == Phase.SUBMISSION:
            return len(self.test_data)
        raise RuntimeError(f"Unknown phase: {self.phase}")

    def __getitem__(self, index) -> Any:
        if self.phase == Phase.TRAIN:
            raise NotImplementedError()
        elif self.phase == Phase.TEST:
            raise NotImplementedError()
        raise RuntimeError(f"Unknown phase: {self.phase}")

    # phase change functions
    def to_train(self):
        self.phase = Phase.TRAIN

    def to_valid(self):
        self.phase = Phase.VALID

    def to_test(self):
        self.phase = Phase.TEST

    def to_dev(self):
        self.phase = Phase.DEV

    def to_submission(self):
        self.phase = Phase.SUBMISSION

    def to(self, phase: Phase):
        if phase == Phase.TRAIN:
            self.to_train()
        elif phase == Phase.VALID:
            self.to_valid()
        elif phase == Phase.TEST:
            self.to_test()
        elif phase == Phase.DEV:
            self.to_dev()
        elif phase == Phase.SUBMISSION:
            self.to_submission()

    @staticmethod
    def collate_fn():
        pass
