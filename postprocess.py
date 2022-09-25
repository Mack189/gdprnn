
import os
import argparse
import numpy as np
from mir_eval.separation import bss_eval_sources
from src.data.data import DatasetGenerator
from src.models.loss import myloss
import mindspore
import mindspore.dataset as ds
import mindspore.ops as ops
from mindspore import Tensor

parser = argparse.ArgumentParser('Evaluate separation performance using TasNet')
parser.add_argument('--bin_path', type=str,
                    default=r"/home/heu_MEDAI/liwenjie/svoice-main-mindspore/outputs/exp_/half1_7007_lwj.ckpt",
                    help='Path to model file created by training')
parser.add_argument('--test_dir', type=str,
                    default=r"/home/liwenjie/svoice-main-ascend/egs/tr",
                    help='directory including mix.json, s1.json and s2.json')
parser.add_argument('--cal_sdr', type=int, default=1,
                    help='Whether calculate SDR, add this option because calculation of SDR is very slow')
parser.add_argument('--use_cuda', type=int, default=0,
                    help='Whether use GPU')
parser.add_argument('--sample_rate', default=8000, type=int,
                    help='Sample rate')
parser.add_argument('--batch_size', default=2, type=int,
                    help='Batch size')
parser.add_argument('--segment', default=4, type=int,
                    help='The hidden size of RNN')

def evaluate(args, list1):
    total_SISNRi = 0
    total_SDRi = 0
    total_cnt = 0

    # Load data
    tt_dataset = DatasetGenerator(args.test_dir, args.batch_size,
                                  sample_rate=args.sample_rate, segment=args.segment)
    tt_loader = ds.GeneratorDataset(tt_dataset, ["mixture", "lens", "sources"], shuffle=False)
    tt_loader = tt_loader.batch(batch_size=2)

    i = 0
    for data in tt_loader.create_dict_iterator():
        padded_mixture = data["mixture"]
        mixture_lengths = data["lens"]
        padded_source = data["sources"]
        padded_mixture = ops.Cast()(padded_mixture, mindspore.float32)
        padded_source = ops.Cast()(padded_source, mindspore.float32)
        mixture_lengths_with_list = mixture_lengths.asnumpy().tolist()
        estimate_source = list1[i]
        i += 1

        my_loss = myloss()
        _, _, estimate_source, reorder_estimate_source = \
            my_loss(padded_source, estimate_source, mixture_lengths)
        mixture = remove_pad(padded_mixture, mixture_lengths_with_list)
        source = remove_pad(padded_source, mixture_lengths_with_list)
        # NOTE: use reorder estimate source
        estimate_source = remove_pad(reorder_estimate_source,
                                     mixture_lengths_with_list)
        # for each utterance
        for mix, src_ref, src_est in zip(mixture, source, estimate_source):
            print("Utt", total_cnt + 1)
            # Compute SDRi
            if args.cal_sdr:
                avg_SDRi = cal_SDRi(src_ref, src_est, mix)
                total_SDRi += avg_SDRi
                print("\tSDRi={0:.2f}".format(-avg_SDRi))
            # Compute SI-SNRi
            avg_SISNRi = cal_SISNRi(src_ref, src_est, mix)
            print("\tSI-SNRi={0:.2f}".format(-avg_SISNRi))
            total_SISNRi += avg_SISNRi
            total_cnt += 1
    if args.cal_sdr:
        print("Average SDR improvement: {0:.2f}".format(total_SDRi / total_cnt))
    print("Average SISNR improvement: {0:.2f}".format(total_SISNRi / total_cnt))


def cal_SDRi(src_ref, src_est, mix):
    """Calculate Source-to-Distortion Ratio improvement (SDRi).
    NOTE: bss_eval_sources is very very slow.
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SDRi
    """
    src_anchor = np.stack([mix, mix], axis=0)
    sdr, _, _, _ = bss_eval_sources(src_ref, src_est)
    sdr0, _, _, _ = bss_eval_sources(src_ref, src_anchor)
    avg_SDRi = ((sdr[0]-sdr0[0]) + (sdr[1]-sdr0[1])) / 2
    return avg_SDRi


def cal_SISNRi(src_ref, src_est, mix):
    """Calculate Scale-Invariant Source-to-Noise Ratio improvement (SI-SNRi)
    Args:
        src_ref: numpy.ndarray, [C, T]
        src_est: numpy.ndarray, [C, T], reordered by best PIT permutation
        mix: numpy.ndarray, [T]
    Returns:
        average_SISNRi
    """
    sisnr1 = cal_SISNR(src_ref[0], src_est[0])
    sisnr2 = cal_SISNR(src_ref[1], src_est[1])
    sisnr1b = cal_SISNR(src_ref[0], mix)
    sisnr2b = cal_SISNR(src_ref[1], mix)
    avg_SISNRi = ((sisnr1 - sisnr1b) + (sisnr2 - sisnr2b)) / 2
    return avg_SISNRi


def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calculate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr


def remove_pad(inputs, inputs_lengths):
    """
    Args:
        inputs: torch.Tensor, [B, C, T] or [B, T], B is batch size
        inputs_lengths: torch.Tensor, [B]
    Returns:
        results: a list containing B items, each item is [C, T], T varies
    """
    results = []
    dim = inputs.ndim
    if dim == 3:
        C = inputs.shape[1]
    for i, input1 in enumerate(inputs):
        if dim == 3:
            results.append(input1[:, :inputs_lengths[i]].view(C, -1).asnumpy())
        elif dim == 2:
            results.append(input1[:inputs_lengths[i]].view(-1).asnumpy())
    return results

if __name__ == "__main__":
    arg = parser.parse_args()
    dataset_path = arg.bin_path
    audio_files = os.listdir(dataset_path)
    audio_files = sorted(audio_files, key=lambda x: int(os.path.splitext(x)[0]))
    list0 = []
    for f in audio_files:
        f_name = os.path.join(dataset_path, f.split('.')[0] + '.bin')
        logits = np.fromfile(f_name, np.float32).reshape(1, 2, 32000)
        logits = Tensor(logits)
        list0.append(logits)
    evaluate(arg, list0)
