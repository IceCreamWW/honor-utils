import argparse
import random
import numpy as np
import torch
import re
import os
import kaldiio
import soundfile as sf
from tqdm import tqdm
from typing import Union, List, Tuple

# python add_noise.py --wav-scp /mnt/lustre/sjtu/home/syz24/myAna/ww089/wenet/examples/wenetspeech/s0/data/train_m/wav.scp --segments /mnt/lustre/sjtu/home/syz24/myAna/ww089/wenet/examples/wenetspeech/s0/data/train_m/segments --noise-scps /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/kaldi/music.scp /mnt/lustre/sjtu/home/ww089/fairseqs/fairseq-220604/examples/hubert/data/musan/kaldi/noise.scp --output /mnt/lustre/sjtu/home/syz24/myAna/ww089/data/wenet/train_m_musan_music_noise_snr_0_10db_1_noise


def get_random_chunk(data: torch.FloatTensor, chunk_len: int) -> torch.FloatTensor:
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(data)
    data_shape = data.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        data = data[chunk_start : chunk_start + chunk_len]
        # re-clone the data to avoid memory leakage
        if type(data) == torch.Tensor:
            data = data.clone()
        else:  # np.array
            data = data.copy()
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (repeat_factor, 1)
        if type(data) == torch.Tensor:
            data = data.repeat(repeat_shape)
        else:  # np.array
            data = np.tile(data, repeat_shape)
        data = data[:chunk_len]

    return data


def read_audio(path, start=0, stop=None):
    if re.match(r".*\.ark:\d+", path):  # kaldi ark style audio path
        sample_rate, wav = kaldiio.load_mat(path)
        if stop is None:
            wav = wav[int(start * sample_rate) :]
        else:
            wav = wav[int(start * sample_rate) : int(stop * sample_rate)]
    else:
        if stop is None:
            wav, sample_rate = sf.read(path, start=int(start * 16000))
        else:
            wav, sample_rate = sf.read(
                path, start=int(start * 16000), stop=int(stop * 16000)
            )
    assert sample_rate == 16000
    wav = torch.from_numpy(wav)
    return wav, sample_rate


def norm_wav(wav: Union[np.ndarray, torch.Tensor]):
    wav = wav / (wav.abs().max() + 1e-5)
    return wav

def additive_noise(
    audio: torch.FloatTensor,
    noises: List[np.ndarray],
    noise_snr: Tuple[int, int],
    num_noise: Tuple[int, int],
):

    audio = norm_wav(audio)
    audio_db = 10 * ((audio ** 2).mean() + 1e-4).log10()
    audio_len = audio.shape[0]

    selected_noises = random.sample(
        noises, random.randint(num_noise[0], num_noise[1])
    )

    noise_list = []
    for noise in selected_noises:
        # noise, noise_sr = read_audio(path)
        # assert noise_sr == 16000
        noise = get_random_chunk(noise, audio_len)
        noise = norm_wav(noise)

        snr = random.uniform(noise_snr[0], noise_snr[1])
        noise_db = 10 * ((noise ** 2).mean() + 1e-4).log10()
        noise_list.append(np.sqrt(10 ** ((audio_db - noise_db - snr) / 10)) * noise)

    audio_aug = torch.stack(noise_list, axis=0).sum(axis=0) + audio
    audio_aug = norm_wav(audio_aug)

    return audio_aug

# def additive_noise(
#     audio: torch.FloatTensor,
#     noise_paths: List[str],
#     noise_snr: Tuple[int, int],
#     num_noise: Tuple[int, int],
# ):
# 
#     audio = norm_wav(audio)
#     audio_db = 10 * ((audio ** 2).mean() + 1e-4).log10()
#     audio_len = audio.shape[0]
# 
#     selected_noise_paths = random.sample(
#         noise_paths, random.randint(num_noise[0], num_noise[1])
#     )
# 
#     noise_list = []
#     for path in selected_noise_paths:
#         noise, noise_sr = read_audio(path)
#         assert noise_sr == 16000
#         noise = get_random_chunk(noise, audio_len)
#         noise = norm_wav(noise)
# 
#         snr = random.uniform(noise_snr[0], noise_snr[1])
#         noise_db = 10 * ((noise ** 2).mean() + 1e-4).log10()
#         noise_list.append(np.sqrt(10 ** ((audio_db - noise_db - snr) / 10)) * noise)
# 
#     audio_aug = torch.stack(noise_list, axis=0).sum(axis=0) + audio
#     audio_aug = norm_wav(audio_aug)
# 
#     return audio_aug


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav-scp", type=str)
    parser.add_argument("--segments", type=str, default=None)
    parser.add_argument("--noise-scps", type=str, nargs="+")
    parser.add_argument("--output", type=str)
    parser.add_argument(
        "--snr",
        type=lambda s: [float(_) for _ in s.split(",")],
        default=[0, 10],
        help="--snr 0,10",
    )
    parser.add_argument(
        "--num-noise",
        type=lambda s: [int(_) for _ in s.split(",")],
        default=[1, 1],
        help="--num-noise 1,1",
    )
    args = parser.parse_args()

    assert len(args.snr) == 2 and args.snr[1] >= args.snr[0]
    assert len(args.num_noise) == 2 and args.num_noise[1] >= args.num_noise[0]

    noises = []
    for noise_scp in args.noise_scps:
        with open(noise_scp) as fp:
            for line in fp:
                noise_path = line.strip().split()[1]
                noise = read_audio(noise_path)[0]
                noises.append(noise)

    wav_dir = os.path.join(args.output, "audio")
    os.makedirs(wav_dir, exist_ok=True)

    if not args.segments:
        num_lines = sum(1 for line in open(args.wav_scp))
        with open(args.wav_scp) as ifp, open(
            os.path.join(args.output, "wav.scp"), "w"
        ) as ofp:
            for line in tqdm(ifp, total=num_lines):
                utt_id, path = line.strip().split()
                wav, sr = read_audio(path)
                wav_aug = additive_noise(
                    wav, noises, noise_snr=args.snr, num_noise=args.num_noise,
                )
                wav_path = os.path.abspath(os.path.join(wav_dir, f"{utt_id}.wav"))
                sf.write(wav_path, wav_aug.numpy(), 16000)
                ofp.write(f"{utt_id} {wav_path}\n")
    else:
        utt2audio_path = {}
        with open(args.wav_scp) as fp:
            for line in fp:
                utt_id, wav_path = line.strip().split()
                utt2audio_path[utt_id] = wav_path
        num_lines = sum(1 for line in open(args.segments))

        spk1_dir = os.path.join(args.output, "spk1")
        os.makedirs(spk1_dir, exist_ok=True)
        with open(args.segments) as ifp, open(
            os.path.join(args.output, "wav.scp"), "w"
        ) as wav_ofp, open(os.path.join(args.output, "spk1.scp"), "w") as spk1_ofp:
            for line in tqdm(ifp, total=num_lines):
                seg_id, utt_id, start, stop = line.strip().split()
                start, stop = float(start), float(stop)
                wav, sr = read_audio(utt2audio_path[utt_id], start=start, stop=stop)
                wav_aug = additive_noise(
                    wav, noises, noise_snr=args.snr, num_noise=args.num_noise,
                )

                wav_path = os.path.abspath(os.path.join(wav_dir, f"{seg_id}.wav"))
                sf.write(wav_path, wav_aug.numpy(), 16000)
                wav_ofp.write(f"{seg_id} {wav_path}\n")

                spk1_path = os.path.abspath(os.path.join(spk1_dir, f"{seg_id}.wav"))
                sf.write(spk1_path, wav.numpy(), 16000)
                spk1_ofp.write(f"{seg_id} {spk1_path}\n")
