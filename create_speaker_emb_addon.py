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

import torch
import msgpack

from speechbrain.pretrained import EncoderClassifier


def parse_args():
    ap = argparse.ArgumentParser(description="Creates speaker embedding addon compatible with balacoon_backend")
    ap.add_argument("--out", required=True, help="Path to put created addon to")
    ap.add_argument("--work-dir", default="./work_dir", help="Directory to put intermediate files to")
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Produces speaker embeddings given
        audio.

        Parameters
        ----------
        x: torch.Tensor
            (1 x samples_num) - audio

        Returns
        -------
        y: torch.Tensor
            (1 x 192) - fixed dimension speaker representation
        """
        # x_len shows which percent of sequence in each batch to use
        # we hardcode to use whole sequence for every element of the batch
        x_len = torch.ones(x.shape[0], dtype=torch.float, device=x.device)
        feats = self._feat(x)
        y = self._emb(feats, x_len)  # 1 x 1 x 192
        return torch.squeeze(y, dim=1)  # 1 x 192


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.work_dir, exist_ok=True)
    device = torch.device("cpu")

    # example of input: 1 sec of 16khz audio
    audio = torch.randn(1, 16000, dtype=torch.float)
    # load the speaker embeddings extractor
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # create extractor
    extractor = SpeakerEmbeddingsExtractor(classifier.mods.compute_features, classifier.mods.embedding_model)
    logging.info("Tracing speaker embedding extractor")
    script = torch.jit.trace(extractor, audio)
    traced_path = os.path.join(args.work_dir, "spkr_emb_extractor.jit")
    script.save(traced_path)

    logging.info("Creating speaker embedding addon")
    addon = {
        "id": "speaker_embedding",
        "sampling_rate": 16000,
        "dimension": 192,
    }
    with open(traced_path, "rb") as fp:
        addon["extractor"] = fp.read()
    with open(args.out, "wb") as fp:
        msgpack.dump([addon], fp)
    logging.info("Saved addon for balacoon_backend to {}".format(args.out))


if __name__ == "__main__":
    main()

