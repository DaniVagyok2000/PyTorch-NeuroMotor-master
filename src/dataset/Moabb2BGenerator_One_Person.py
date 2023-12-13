import numpy as np
import moabb
from moabb.datasets import BNCI2014_004
from moabb.evaluations import CrossSessionEvaluation
from moabb.paradigms import MotorImagery
import importlib
from typing import Tuple, List, Union

import mne
mne.set_log_level('error')

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import torch

from config.default import cfg


class Moabb2BGenerator(Dataset):
    def __init__(self, subject_id,device, runs,train_data: bool = False,return_subject_id: bool = False) -> None:
        self.scaler = StandardScaler()
        self.subject_id = subject_id
        self.runs = runs
        self.return_subject_id = return_subject_id
        self.train_data = train_data
        self.load_config("default")
        self.data = self.load_data()
        self.raw_runs = self.preprocess_data(self.data)
        self.create_epochs(self.raw_runs)

        self.time_steps = self.X.shape[-1]
        self.channels = self.X.shape[-2]

        self.device = device

        self.split_by_runs()

        self.format_data()
            

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

    def load_data(self):
        # Load the BCI Competition 2B dataset
        dataset = BNCI2014_004()        
        original_data = dataset.get_data([self.subject_id])
        return original_data


    def preprocess_data(self,original_data):
        raw_runs = []
        for subj in original_data:
            subject_data = original_data[subj]
            if self.train_data:
                raw_runs.extend(list(subject_data['0train'].values()))
                raw_runs.extend(list(subject_data['1train'].values()))
                raw_runs.extend(list(subject_data['2train'].values()))
            else:
                raw_runs.extend(list(subject_data['3test'].values()))
                raw_runs.extend(list(subject_data['4test'].values()))

        mapping = {1:'left_hand', 2:'right_hand'}

        for raw in raw_runs:
            ch_types = {ch: 'eeg' if ch != 'stim' else 'stim' for ch in raw.ch_names}
            raw.set_channel_types(ch_types)
            events = mne.find_events(raw, stim_channel='stim')
            
            annot_from_events = mne.annotations_from_events(events, event_desc=mapping, sfreq=raw.info['sfreq'])
            raw.set_annotations(annot_from_events)
            raw.drop_channels(['stim'])

        motor_cortex_channels = ['C3', 'Cz', 'C4']
        
        for run in raw_runs:
            run = run.resample(self.target_freq, npad="auto")
            run = run.set_eeg_reference('average', projection=True)
            run = run.pick_channels(motor_cortex_channels)
            run = run.filter(l_freq=self.low_freq, h_freq=self.high_freq)

        return raw_runs
    
    def create_epochs(self, raw_runs) -> None:
        self.X = np.array([])
        self.y = np.array([])
        for raw in raw_runs:
            events, event_ids = mne.events_from_annotations(raw)
            epoch = mne.Epochs(
                raw,
                events=events,
                event_id=event_ids,
                tmin=self.tmin,  
                tmax=self.tmax,  
                baseline=self.baseline, 
                preload=True,
            )
           
            epoch = epoch.crop(tmin=self.baseline[-1], tmax=self.tmax)  
            data,labels = self.extract_data(epoch=epoch)            

            if self.X.size == 0:
                self.X = data
                self.y = labels
            else:
                self.X = np.concatenate((self.X, data), axis=0)
                self.y = np.concatenate((self.y, labels), axis=0)

    def extract_data(self,epoch) -> None:
        X = epoch.get_data()     

        if self.normalize:
            X = self.do_normalize(X)

        y = epoch.events[:, -1]
        y -= 1  # start at 0

        return X,y

    def do_normalize(self,X) -> None:
        orig_shape =X.shape
        X = X.reshape(X.shape[0], -1)
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        X = X.reshape(orig_shape)        

        return X    
    
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
        self.X = self.X.reshape(-1, self.X.shape[2], self.X.shape[3])
        self.y = self.y.reshape(-1)

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

        
