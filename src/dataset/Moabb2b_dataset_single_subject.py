import os
import numpy as np
import importlib
import mne
from typing import Tuple, List, Union

mne.set_log_level("error")

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from moabb.datasets import BNCI2014_004
from sklearn.preprocessing import StandardScaler

MAPPING = {1: "left_hand", 2: "right_hand"}


class MI_Dataset(Dataset):
    def __init__(
        self,
        subject_id: int,
        runs: List[int],
        device: Union[str, torch.device] = "cpu",
        config: str = "default",
        return_subject_id: bool = False,
        verbose: bool = False,
    ):
        """
        Initializes MI_Dataset.

        Args:
            subject_id (int): Subject ID to train on.
            runs (List[int]): List of runs to train on.
            device (Union[str, torch.device], optional): Device to use for data. Defaults to "cpu".
            config (str, optional): Configuration file to use. Defaults to "default".
            verbose (bool, optional): If True, print additional information. Defaults to False.
        """
        self.data_root = "data"
        self.subject_id = subject_id
        self.device = device
        self.runs = runs
        self.return_subject_id = return_subject_id

        self.load_config(config)

        self.load_raw()
        self.apply_preprocess()
        self.create_epochs()

        self.extract_data()
        self.split_by_runs()
        self.format_data()

        self.time_steps = self.X.shape[-1]
        self.channels = self.X.shape[-2]

        if verbose:
            print("#" * 50)
            print("Dataset created:")
            print(f"X --> {self.X.shape} ({self.X.dtype})")
            print(f"y --> {self.y.shape} ({self.y.dtype})")
            print("#" * 50)

    def preprocess_data(self,original_data):
        raw_runs = []
        for subj in original_data:
            subject_data = original_data[subj]
            raw_runs.extend(list(subject_data['0train'].values()))
            raw_runs.extend(list(subject_data['1train'].values()))
            raw_runs.extend(list(subject_data['2train'].values()))
            raw_runs.extend(list(subject_data['3test'].values()))
            raw_runs.extend(list(subject_data['4test'].values()))

        return raw_runs

    def load_data(self):
        # Load the BCI Competition 2B dataset
        dataset = BNCI2014_004()        
        original_data = dataset.get_data([self.subject_id])
        return original_data

    def load_config(self, file: str) -> None:
        cfg = importlib.import_module(f"config.{file}").cfg

        self.target_freq = cfg["preprocessing"]["target_freq"]
        self.low_freq = cfg["preprocessing"]["low_freq"]
        self.high_freq = cfg["preprocessing"]["high_freq"]
        self.average_ref = cfg["preprocessing"]["average_ref"]

        self.baseline = cfg["epochs"]["baseline"]
        self.tmin = cfg["epochs"]["tmin"]
        self.tmax = cfg["epochs"]["tmax"]

        self.normalize = cfg["train"]["normalize"]

    def load_raw(self) -> None:   
        data = self.load_data()

        self.raw = self.preprocess_data(data)
        self.filter_events()
        eog_channels = [
            i for i, ch_name in enumerate(self.raw.ch_names) if "EOG" in ch_name
        ]
        self.raw.drop_channels([self.raw.ch_names[ch_idx] for ch_idx in eog_channels])

    def filter_events(self) -> None:
        events, _ = mne.events_from_annotations(self.raw)
        annot_from_events = mne.annotations_from_events(
            events, event_desc=MAPPING, sfreq=self.raw.info["sfreq"]
        )

        self.raw.set_annotations(annot_from_events)

    def apply_preprocess(self) -> None:
        if self.target_freq:
            self.raw = self.raw.resample(self.target_freq, npad="auto")
        if self.average_ref:
            self.raw = self.raw.set_eeg_reference("average", projection=True)

        self.raw = self.raw.filter(l_freq=self.low_freq, h_freq=self.high_freq)

    def create_epochs(self) -> None:
        events, event_ids = mne.events_from_annotations(self.raw)

        self.epochs = mne.Epochs(
            self.raw,
            events=events,
            event_id=event_ids,
            tmin=self.tmin,
            tmax=self.tmax,
            baseline=self.baseline,
            preload=True,
        )

        self.epochs = self.epochs.crop(tmin=self.baseline[-1], tmax=self.tmax)

        del self.raw

    def extract_data(self) -> None:
        self.X = self.epochs.get_data()

        if self.normalize:
            self.do_normalize()

        self.y = self.epochs.events[:, -1]
        self.y -= 1  # start at 0

    def do_normalize(self) -> None:
        orig_shape = self.X.shape
        self.X = self.X.reshape(self.X.shape[0], -1)
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        self.X = self.X.reshape(orig_shape)

    def split_by_runs(self) -> None:
        X_by_runs = []
        y_by_runs = []

        for index in range(0, int(self.X.shape[0] // 48)):
            X_by_runs.append(self.X[index * 48 : (index + 1) * 48])
            y_by_runs.append(self.y[index * 48 : (index + 1) * 48])

        self.runs_features = np.array(X_by_runs)
        self.runs_labels = np.array(y_by_runs)

        self.X = self.runs_features[self.runs]
        self.y = self.runs_labels[self.runs]

    def format_data(self) -> None:
        # Remove Run dimension
        print("Before",self.X.shape)
        self.X = self.X.reshape(-1, self.X.shape[2], self.X.shape[3])
        self.y = self.y.reshape(-1)
        print("After",self.X.shape)
        print("label", self.y.shape)

        self.X = torch.from_numpy(self.X).float()
        self.y = torch.from_numpy(self.y).long()

        self.X = self.X.to(self.device)
        self.y = self.y.to(self.device)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.return_subject_id:
            return ((self.X[idx], torch.tensor(self.subject_id-1, dtype=torch.int64)), self.y[idx])
        else:
         return (self.X[idx],  self.y[idx])
