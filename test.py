"""
Copyright 2022 Balacoon

Test of traced ECAPA TDNN model.
Shows that embeddings extracted from files
with same speaker are closer than embeddings
extracted from files with same text.
"""

import torch
import soundfile
import numpy as np
from scipy import spatial


import torchaudio
from speechbrain.pretrained import EncoderClassifier
import matplotlib.pylab as plt


def main():
    model = torch.jit.load("traced_gpu/ecapa_speaker_embedding.jit").to(torch.device("cuda"))

    # prepare data for inference
    audio_paths = ["slt_arctic_a0001", "slt_arctic_a0002", "rms_arctic_a0001"]
    audio = []
    for path in audio_paths:
        data, sr = soundfile.read(path + ".wav", dtype="int16")
        assert sr == 16000, "model expects 16khz"
        audio.append(data)
    audio_len = [x.shape[0] for x in audio]
    max_audio_len = max(audio_len)
    audio = [np.concatenate([x, np.zeros((max_audio_len - x.shape[0],), dtype=np.int16)]) for x in audio]
    audio = torch.tensor(audio).cuda()
    audio_len = torch.tensor(audio_len).int().cuda()

    # run inference, print pariwise distance
    embs = model(audio, audio_len).detach().cpu().numpy()
    for i in range(len(audio_paths)):
        a = embs[i]
        for j in range(i + 1, len(audio_paths)):
            b = embs[j]
            dist = spatial.distance.cosine(a, b)
            print("{} < - > {}: {}".format(audio_paths[i], audio_paths[j], dist))

    # run inference with untraced model
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")
    signal, fs = torchaudio.load('slt_arctic_a0001.wav')
    embeddings = classifier.encode_batch(signal)
    emb_orig = torch.squeeze(embeddings).cpu().detach().numpy()
    plt.plot(embs[1])
    plt.plot(emb_orig)
    plt.show()


if __name__ == "__main__":
    main()
