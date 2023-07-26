import sys
import os

sys.path.append("/data/dev/cpp/tinfer/tinfer/")
print(sys.path)
from tools.model import Model

max_seq_len = 128
model = Model("./config.json")
model.add_weight("wpe", [max_seq_len, model.config.hidden_units])
model.add_weight("wte", [model.config.vocab_size, model.config.hidden_units])
model.add_weight("final_layernorm.bias", [model.config.hidden_units])
model.add_weight("final_layernorm.weight", [model.config.hidden_units])
model.add_layer_weight("attention.dense.bias", [model.config.hidden_units])
model.add_layer_weight(
    "attention.dense.weight.0", [model.config.hidden_units, model.config.hidden_units]
)
model.add_layer_weight(
    "attention.query_key_value.bias.0", [3, model.config.hidden_units]
)
model.add_layer_weight(
    "attention.query_key_value.weight.0",
    [model.config.hidden_units, 3 * model.config.hidden_units],
)
model.add_layer_weight(
    "input_layernorm.bias",
    [model.config.hidden_units],
)
model.add_layer_weight(
    "input_layernorm.weight",
    [model.config.hidden_units],
)
model.add_layer_weight(
    "mlp.dense_4h_to_h.bias",
    [model.config.hidden_units],
)
model.add_layer_weight(
    "mlp.dense_4h_to_h.weight.0",
    [model.config.inter_size, model.config.hidden_units],
)
model.add_layer_weight(
    "mlp.dense_h_to_4h.bias.0",
    [model.config.inter_size],
)
model.add_layer_weight(
    "mlp.dense_h_to_4h.weight.0",
    [model.config.hidden_units, model.config.inter_size],
)
model.add_layer_weight(
    "post_attention_layernorm.bias",
    [model.config.hidden_units],
)
model.add_layer_weight(
    "post_attention_layernorm.weight",
    [model.config.hidden_units],
)

model.dump_weight_config("./model_weights.json")
model.add_op("prepare_inputs")
model.add_op("tile_prompt_inputs")
model.add_op("inputs_embedding_lookup_pos_encoding")
model.add_op("build_decoder_attention_mask")
model.dump_ops("./model_ops.json")
