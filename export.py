
import argparse
import numpy as np
from src.models.swave import SWave
from mindspore.train.serialization import export
from mindspore import Tensor, context
from mindspore.train.serialization import load_checkpoint, load_param_into_net


parser = argparse.ArgumentParser()
parser.add_argument('--N', default=128, type=int,
                    help='The number of expected features in the input')
parser.add_argument('--L', default=8, type=int,
                    help='kernel sizes')
parser.add_argument('--H', default=128, type=int,
                    help='The hidden size of RNN')
parser.add_argument('--R', default=6, type=int,
                    help='Model layers')
parser.add_argument('--C', default=2, type=int,
                    help='Maximum number of speakers')
parser.add_argument('--sr', default=8000, type=int,
                    help='Sample rate of audio file')
parser.add_argument('--input_normalize', default=False, type=bool,
                    help='Normalize or not')
parser.add_argument('--ckpt_path', default="/home/heu_MEDAI/1_gdprnn.ckpt",
                    help='Path to model file created by training')

def export_gdprnn():
    """ export """
    args = parser.parse_args()
    net = SWave(args.N, args.L, args.H, args.R, args.C, args.sr, args.segment, input_normalize=False)

    param_dict = load_checkpoint(args.ckpt_path)
    load_param_into_net(net, param_dict)
    input_data = Tensor(np.random.uniform(0.0, 1.0, size=[1, 32000]).astype(np.float32))
    export(net, input_data, file_name='SWave', file_format='MINDIR')
    print("export success")

if __name__ == '__main__':
    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=4)
    export_gdprnn()
