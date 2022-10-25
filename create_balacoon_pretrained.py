"""
Copyright 2022 Balacoon

Script to export pretrained speaker embedding model as an addon
compatible with balacoon_backend.
Exporting ECAPA_TDNN model:
https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
"""
import argparse
import logging
import os
from typing import Tuple

import torch
import torchaudio
import msgpack
from scipy import spatial

from speechbrain.pretrained import EncoderClassifier


def parse_args():
    ap = argparse.ArgumentParser(description="Creates speaker embedding traced model and addon compatible with balacoon_backend")
    ap.add_argument("--out-dir", required=True, help="Directory to put traced model and addon")
    ap.add_argument("--use-gpu", action="store_true", help="Whether to trace on gpu")
    args = ap.parse_args()
    return args


class SpeakerEmbeddingsExtractor(torch.nn.Module):
    """
    Wraps encoder classifier from speechbrain, ommiting feature normalization
    (not used in speechbrain/spkrec-ecapa-voxceleb) and squeezing
    the output speaker embedding
    """
    def __init__(self, feat, emb):
        super().__init__()
        self._feat = feat
        self._emb = emb

    def forward(self, x: torch.Tensor, x_len: torch.Tensor) -> torch.Tensor:
        """
        Produces speaker embeddings given
        audio.

        Parameters
        ----------
        x: torch.Tensor
            (batch x samples_num) - audio
        x_len: torch.Tensor
            (batch) - real len of samples in each audio from batch

        Returns
        -------
        y: torch.Tensor
            (batch x 192) - fixed dimension speaker representation
        """
        # x_len shows which percent of sequence in each batch to use
        # we hardcode to use whole sequence for every element of the batch
        x_percent = x_len.float() / torch.ones(x.size(0), dtype=torch.float, device=x.device) * x.size(1)
        feats = self._feat(x)
        y = self._emb(feats, x_percent)  # 1 x 1 x 192
        return torch.squeeze(y, dim=1)  # 1 x 192


def load_data(path: str, use_gpu: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    loads audio data from a wav file
    """
    audio, sr = torchaudio.load(path)
    if use_gpu:
        audio = audio.cuda()
    audio_len = torch.tensor([audio.size(1)], dtype=torch.int, device=audio.device)
    assert sr == 16000, "Sampling rate for speaker embedding extraction is 16kHz"
    return audio, audio_len


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if args.use_gpu else "cpu")

    # load example of audio
    audio, audio_len = load_data("slt_arctic_a0001.wav", args.use_gpu)
    # load the speaker embeddings extractor
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # create extractor
    extractor = SpeakerEmbeddingsExtractor(classifier.mods.compute_features,
        classifier.mods.embedding_model).to(device)
    logging.info("Tracing speaker embedding extractor")
    traced = torch.jit.trace(extractor, [audio, audio_len])
    traced_path = os.path.join(args.out_dir, "ecapa_tdnn_spkr_emb.jit")
    traced.save(traced_path)

    logging.info("Creating speaker embedding addon")
    addon = {
        "id": "speaker_embedding",
        "sampling_rate": 16000,
        "dimension": 192,
    }
    with open(traced_path, "rb") as fp:
        addon["extractor"] = fp.read()
    addon_path = os.path.join(args.out_dir, "ecapa_tdnn_spkr_emb.addon")
    with open(addon_path, "wb") as fp:
        msgpack.dump([addon], fp)
    logging.info("Saved addon for balacoon_backend to {}".format(addon_path))

    # check that embeddings for audio from same speaker are close
    # and from embedding from a different speaker
    wav_files = ["slt_arctic_a0001.wav", "slt_arctic_a0002.wav", "rms_arctic_a0001.wav"]
    embs = []
    for path in wav_files:
        audio, audio_len = load_data(path, args.use_gpu)
        emb = traced(audio, audio_len).detach().cpu().numpy()
        embs.append(emb)
    for i, a in enumerate(embs):
        for j in range(i + 1, len(embs)):
            b = embs[j]
            dist = 1 - spatial.distance.cosine(a, b)
            logging.info("{} < - > {}: {}".format(wav_files[i], wav_files[j], dist))



if __name__ == "__main__":
    main()

