"""
Copyright 2022 Balacoon

Script to trace pretrained inhouse speaker embedding model,
compatible with balacoon data preparation
"""
import argparse
import logging
import os

import matplotlib.pylab as plt
import soundfile
import torch

from speechbrain.pretrained import EncoderClassifier


class SpeakerEmbeddingExtractor(torch.nn.Module):
    """
    Wrapper for ECAPA TDNN speaker embedding extractor by SpeechBrain
    Combines feature extraction and model execution. Omits feature normalization
    since it's not used in speaker embedding, squeezes output embedding to have
    shape (batch_size x 192). Finally, converts length of input sequences within batch
    to percent values, as explained here:
    https://github.com/speechbrain/speechbrain/blob/develop/speechbrain/pretrained/interfaces.py#L916.
    Unfortunately implementation of TDNN by Speechbrain contains hardcoded types,
    thus for now tracing is only possible in full precision.
    """
    def __init__(self, feat, norm, emb):
        super().__init__()
        self._feat = feat
        self._norm = norm
        self._emb = emb

    def forward(self, audio: torch.Tensor, audio_len: torch.Tensor) -> torch.Tensor:
        """
        extracts speaker embeddings from provided audio. wrapper around speechbrain
        modules unifies interface with other pre-trained speech models.

        Parameters
        ----------
        audio: torch.Tensor
            (batch_size x samples_num) - audio samples in int16 in range (-INT_MAX; INT_MAX)
        audio_len: torch.Tensor
            (batch_size,) - input that specifies actual length of audio within batch.
            Should be int32, stored on CPU regardless of config (cuda, half)

        Returns
        -------
        speaker_emb: torch.Tensor
            (batch_size x 192) - speaker representation(s) extracted for input audio.
        """
        # convert audio from short to float
        audio = audio.type(torch.float32) / 32768.0  # to range (-1, 1)

        # convert audio_len to audio_percent, i.e. instead of specifying how
        # many samples in each sequence within batch - specify how many percent of sequence
        # is valid and how much is padding.
        audio_len = audio_len.type(torch.float32)
        audio_percent = audio_len / (torch.ones_like(audio_len) * torch.max(audio_len))
        feats = self._feat(audio)
        feats = self._norm(feats, audio_percent)
        emb = self._emb(feats, audio_percent)
        return torch.squeeze(emb, dim=1)  # 1 x 192


def parse_args():
    ap = argparse.ArgumentParser(
        description="Traces ECAPA speaker embedding model. Only able to trace in full precision for now"
    )
    ap.add_argument("--audio", required=True, help="Long audio file to collect stats for feats normalization")
    ap.add_argument(
        "--out-dir",
        default="./exported",
        help="Directory to put exported files to",
    )
    ap.add_argument("--cpu", action="store_true", help="If specified, traces on CPU rather than GPU")
    args = ap.parse_args()
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cpu" if args.cpu else "cuda")

    # load the original speaker embeddings extractor
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    # wrap speaker embedding extractor
    extractor = SpeakerEmbeddingExtractor(
        classifier.mods.compute_features,
        classifier.mods.mean_var_norm,
        classifier.mods.embedding_model,
    ).to(device)
    extractor.eval()

    # prepare the input
    audio_real, sample_rate = soundfile.read(args.audio, dtype="int16")
    assert sample_rate == 16000, "Speaker embedding extractor works with 16khz audio"
    audio_len = torch.tensor([audio_real.shape[0]]).int().to(device)
    audio_real = torch.tensor([audio_real]).to(device)

    with torch.no_grad():
        # run inference and check output
        emb = extractor(audio_real, audio_len).detach().cpu().numpy()[0]
        plt.plot(emb)
        plt.show()

        # trace the model and save result
        extractor_traced = torch.jit.trace(extractor, [audio_real, audio_len])
        out_path = os.path.join(args.out_dir, "ecapa_speaker_embedding.jit")
        extractor_traced.save(out_path)
        logging.info("Saved traced ECAPA speaker embedding extractor to {}".format(out_path))


if __name__ == "__main__":
    main()
