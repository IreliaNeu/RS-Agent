import torch.optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils import data
import sys 
import argparse
import json
from tqdm import tqdm


from data.LEVIR_MCI import LEVIRCCDataset
from model.model_encoder_att import Encoder, AttentiveEncoder
from model.model_decoder import DecoderTransformer

import lagent
from utils_tool.utils import *
from utils_tool.metrics import Evaluator
import predict
from lagent import list_tools

print(lagent.__version__),
print(list_tools())

model1 = predict.Change_Perception()
# model2 = predict.Change_Perception()
summary = model1.convert_to_multimodal(
    input_dir_A = r'D:\Dataset\WHU_test6_40_108_109_110\A',
    input_dir_B = r'D:\Dataset\WHU_test6_40_108_109_110\B',
    output_dir = r'D:\Dataset\WHU_test6_40_108_109_110\output_text\text1'
)
print(summary)
summary = model1.convert_to_multimodal(
    input_dir_A = r'D:\Dataset\WHU_test6_40_108_109_110\A',
    input_dir_B = r'D:\Dataset\WHU_test6_40_108_109_110\B',
    output_dir = r'D:\Dataset\WHU_test6_40_108_109_110\output_text\text2'
)
print(summary)