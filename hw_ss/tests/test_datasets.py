import unittest

import torch

from hw_ss.datasets import LibrispeechDataset, CustomDirAudioDataset, CustomAudioDataset
from hw_ss.tests.utils import clear_log_folder_after_use
from hw_ss.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_ss.utils import ROOT_PATH
from hw_ss.utils.parse_config import ConfigParser


class TestDataset(unittest.TestCase):
    def test_librispeech(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean",
                text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser,
                max_text_length=140,
                max_audio_length=13,
                limit=10,
            )
            self._assert_training_example_is_good(ds[0])

    def _assert_training_example_is_good(self, training_example: dict, contains_text=True):

        for field, expected_type in [
            ("audio_target", torch.Tensor),
            ("spectrogram_target", torch.Tensor),
            ("audio_mixed", torch.Tensor),
            ("spectrogram_mixed", torch.Tensor),
            ("audio_ref", torch.Tensor),
            ("spectrogram_ref", torch.Tensor),
            ("audio_path", str)
        ]:
            self.assertIn(field, training_example, f"Error during checking field {field}")
            self.assertIsInstance(training_example[field], expected_type,
                                  f"Error during checking field {field}")
