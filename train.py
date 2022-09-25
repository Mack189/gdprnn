
import os
import argparse
import time
from src.data.preprocess import preprocess
from src.models.swave import SWave
from src.data.data import DatasetGenerator
from src.generatorloss import Generatorloss
from src.trainonestep import TrainOneStep
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
import mindspore.dataset as ds
from mindspore import nn
from mindspore import context
from mindspore.context import ParallelMode
from mindspore.communication.management import init, get_rank, get_group_size

parser = argparse.ArgumentParser()
parser.add_argument('--in-dir', type=str, default=r"/home/work/user-job-dir/inputs/data/",
                    help='Directory path of LS-2mix including tr, cv and tt')
parser.add_argument('--out-dir', type=str, default=r"/home/work/user-job-dir/inputs/data_json",
                    help='Directory path to put output files')
parser.add_argument('--sample-rate', type=int, default=8000,
                    help='Sample rate of audio file')
parser.add_argument('--data_url',
                    help='path to training/inference dataset folder',
                    default='/home/work/user-job-dir/inputs/data/')
parser.add_argument('--train_url',
                    help='Model folder to save/load',
                    default='/home/work/user-job-dir/model/')
parser.add_argument(
    '--device_target',
    type=str,
    default="Ascend",
    choices=['Ascend', 'GPU', 'CPU'],
    help='device where the code will be implemented (default: Ascend)')
parser.add_argument('--segment', type=int, default=4,
                    help='Segment size')
parser.add_argument('--batch_size', type=int, default=6,
                    help='Batch size')
parser.add_argument('--epochs', type=int, default=120,
                    help='Epoch')
parser.add_argument('--device_num', type=int, default=8,
                    help='Device num')
parser.add_argument('--device_id', type=int, default=0,
                    help='Device id')
parser.add_argument('--is_distribute', type=bool, default=True,
                    help='Distribute or not')
parser.add_argument('--data_batch_size', type=int, default=3,
                    help='Data num')
parser.add_argument('--train', type=str, default='/home/work/user-job-dir/inputs/data_json/tr',
                    help='path to training/inference dataset folder')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='Learning rate')
parser.add_argument('--modelArts', default=0, type=int,
                    help='Cload')
parser.add_argument('--continue_train', default=0, type=int,
                    help='Continue from checkpoint model')
parser.add_argument('--model_type', type=str, default='swave')
parser.add_argument('--save_folder', default='output',
                    help='Location to save epoch models')
parser.add_argument('--ckpt_path', default='1_gdprnn.ckpt',
                    help='Path to model file created by training')
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

def train(trainoneStep, dataset, train_dir, obs_train_url, args):
    trainoneStep.set_train()
    trainoneStep.set_grad()
    tr_loader = dataset['tr_loader']
    step = tr_loader.get_dataset_size()

    for epoch in range(args.epochs):

        total_loss = 0
        j = 0
        for data in tr_loader:
            mixture, lens, source = [x for x in data]
            t0 = time.time()
            loss = trainoneStep(mixture, lens, source)
            t1 = time.time()
            if j % 30 == 0:
                print("epoch[{}]({}/{}),loss:{:.4f},stepTime:{}".format(epoch + 1, j+1, step, loss.asnumpy(), t1 - t0))
            j = j + 1
            total_loss += loss
        train_loss = total_loss/j
        print("epoch[{}]:trainAvgLoss:{:.4f}".format(epoch + 1, train_loss.asnumpy()))
        if args.modelArts:
            save_checkpoint_path = train_dir + '/device_' + os.getenv('DEVICE_ID') + '/'
        else:
            save_checkpoint_path = args.save_folder
        if not os.path.exists(save_checkpoint_path):
            os.makedirs(save_checkpoint_path)
        save_ckpt = os.path.join(save_checkpoint_path, '{}_gdprnn.ckpt'.format(epoch + 1))
        save_checkpoint(trainoneStep.network, save_ckpt)
        if args.modelArts:
            mox.file.copy_parallel(train_dir, obs_train_url)
            print("Successfully Upload {} to {}".format(train_dir,
                                                        obs_train_url))

def main(args):
    device_num = int(os.environ.get("RANK_SIZE", 1))
    if device_num == 1:
        is_distributed = 'False'
    elif device_num > 1:
        is_distributed = 'True'

    if is_distributed == 'True':
        print("parallel init", flush=True)
        init()
        rank_id = get_rank()
        context.reset_auto_parallel_context()
        parallel_mode = ParallelMode.DATA_PARALLEL
        rank_size = get_group_size()
        context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True, device_num=args.device_num)
        context.set_auto_parallel_context(parameter_broadcast=True)
        print("Starting traning on multiple devices...")
    else:
        if args.modelArts:
            init()
            rank_id = get_rank()
            rank_size = get_group_size()
        else:
            context.set_context(device_id=args.device_id)
    if args.modelArts:
        import moxing as mox
        home = os.path.dirname(os.path.realpath(__file__))
        obs_data_url = args.data_url
        args.data_url = '/home/work/user-job-dir/inputs/data/'
        train_dir = os.path.join(home, 'checkpoints') + str(rank_id)

        obs_train_url = args.train_url
        if not os.path.exists(train_dir):
            os.mkdir(train_dir)

        mox.file.copy_parallel(obs_data_url, args.data_url)
        print("Successfully Download {} to {}".format(obs_data_url,
                                                      args.data_url))
        preprocess(args)

    net = SWave(args.N, args.L, args.H, args.R, args.C, args.sr, args.segment, input_normalize=False)


    if args.continue_train:
        if args.modelArts:
            home = os.path.dirname(os.path.realpath(__file__))
            ckpt = os.path.join(home, args.ckpt_path)
            params = load_checkpoint(ckpt)
            load_param_into_net(net, params)
        else:
            params = load_checkpoint(args.ckpt_path)
            load_param_into_net(net, params)

    tr_dataset = DatasetGenerator(args.train, args.data_batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    if is_distributed == 'True':
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False, num_shards=rank_size, shard_id=rank_id)
    else:
        tr_loader = ds.GeneratorDataset(tr_dataset, ["mixture", "lens", "sources"],
                                        shuffle=False)
    tr_loader = tr_loader.batch(args.batch_size)
    num_steps = tr_loader.get_dataset_size()
    data = {"tr_loader": tr_loader}

    loss_network = Generatorloss(net)
    milestone = [45 * num_steps, 78 * num_steps, 120 * num_steps]
    learning_rates = [1e-3, 5e-4, 2e-4]
    lr = nn.piecewise_constant_lr(milestone, learning_rates)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr, beta1=0.9, beta2=0.999)
    trainonestepNet = TrainOneStep(loss_network, optimizer, sens=1.0)
    if args.modelArts:
        train(trainonestepNet, data, train_dir, obs_train_url, args)
    else:
        train_dir = '/home/'
        obs_train_url = '/home/'
        train(trainonestepNet, data, train_dir, obs_train_url, args)

if __name__ == '__main__':
    arg = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=arg.device_target)
    main(arg)
