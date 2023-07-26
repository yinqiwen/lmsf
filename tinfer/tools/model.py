import json
from typing import Optional
from typing import List
from types import SimpleNamespace

from dataclasses import dataclass
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class ModelConfig:
    model_type: str
    model_name: str
    head_num: int
    size_per_head: int
    inter_size: int
    max_pos_seq_len: int
    num_layer: int
    vocab_size: int
    start_id: int
    end_id: int
    weight_data_type: str
    tensor_para_size: int
    hidden_units: Optional[int] = None


class Model:
    def __init__(self, json_file):
        self.weight_config = {}
        self.weight_config["common"] = {}
        self.weight_config["layers"] = {}
        with open(json_file) as jfile:
            config_contents = jfile.read()
        self.config = ModelConfig.from_json(config_contents)
        if self.config.hidden_units is None:
            self.config.hidden_units = self.config.head_num * self.config.size_per_head
        self.weight_config["layers"]["count"] = self.config.num_layer
        self.weight_config["layers"]["weights"] = {}
        self.ops = []

    def add_weight(self, name, shape):
        self.weight_config["common"][name] = shape

    def add_layer_weight(self, name, shape):
        self.weight_config["layers"]["weights"][name] = shape

    def add_op(self, name, once=False, is_layer_op = False, args=None):
        op = {}
        op["name"] = name
        op["once"] = once
        op["is_layer_op"] = is_layer_op
        if args is not None:
            op["args"] = args
        self.ops.append(op)

    def dump_weight_config(self, file):
        with open(file, "w") as outfile:
            outfile.write(json.dumps(self.weight_config, sort_keys=True, indent=4))

    def dump_ops(self, file):
        with open(file, "w") as outfile:
            outfile.write(json.dumps(self.ops, sort_keys=True, indent=4))
