import os
import json
import datetime
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

from src.lapis.utils.log import copy_file
from src.lapis.agents.agent import Agent
from src.lapis.logger_cfg import logger

class BasePipeline(ABC):
    def __init__(self, 
                 base_dir, 
                 data_dir, 
                 results_dir, 
                 splits,
                 agent,
                 generate_domain,
                 ground_in_sg=False):
        self.base_dir: str = base_dir
        self.data_dir: str = data_dir
        self.results_dir: str = results_dir
        self.splits: List[str] = splits
        self.agent: Agent = agent
        self.generate_domain: bool = generate_domain
        self.ground_in_sg: bool = ground_in_sg

        self._name: str = "BasePipeline"
        

        
        self.experiment_name: str = self._construct_experiment_name()
        self.timestamp: str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.current_phase: Optional[str] = None
    
    @property
    def name(self):
        return type(self).__name__
    
    def _construct_experiment_name(self):
        name = f"{self.name}_{self.agent.name}"
        if self.generate_domain:
            name += "_gendomain"
        if self.ground_in_sg:
            name += "_ground"
        return name
    
    def run(self):
        results_dir = os.path.join(self.results_dir, self.experiment_name, self.timestamp)
        os.makedirs(results_dir, exist_ok=True)
        
        for task_idx, task_name in enumerate(self.splits, 1):
            print(f"\n[Pipeline] Task {task_idx}/{len(self.splits)}: {task_name}")
            self._process_task(task_name, results_dir)

    @abstractmethod
    def _process_task(self, task_name, results_dir):
        pass

    @abstractmethod
    def _initialize_csv(self, csv_filepath):
        pass
