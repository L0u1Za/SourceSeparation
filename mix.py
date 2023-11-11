import argparse
import os
from hw_ss.mixer.mixer import LibriSpeechSpeakerFiles, MixtureGenerator

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Dataset Mixer")
    args.add_argument('-p', '--path',
        default=None,
        type=str,
        help="dataset path (default: None)",
    )
    args.add_argument('-o', '--output',
        default=None,
        type=str,
        help="output mixture path (default: None)",
    )
    args.add_argument('-t', '--train',
        default=True,
        type=bool,
        help="True if it is train, false if test (default: None)",
    )
    args = args.parse_args()
    path, o_path, train = args.path, args.output, args.train

    speakers = [el.name for el in os.scandir(path)]
    speakers_files = [LibriSpeechSpeakerFiles(i, path, audioTemplate='*.flac') for i in speakers]
    mixer = MixtureGenerator(speakers_files, o_path, test=not(train))
    mixer.generate_mixes(snr_levels=[-5, 5], num_workers=2, update_steps=100, trim_db=20, vad_db=20, audioLen=3)