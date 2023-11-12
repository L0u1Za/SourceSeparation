import unittest

from tqdm import tqdm

from hw_ss.collate_fn.collate import collate_fn
from hw_ss.datasets import LibrispeechDataset
from hw_ss.tests.utils import clear_log_folder_after_use
from hw_ss.utils.object_loading import get_dataloaders
from hw_ss.utils.parse_config import ConfigParser


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            ds = LibrispeechDataset(
                "dev-clean", text_encoder=config_parser.get_text_encoder(),
                config_parser=config_parser
            )

            batch_size = 3
            batch = collate_fn([ds[i] for i in range(batch_size)])

            self.assertIn("audio_target", batch)  # torch.tensor
            self.assertIn("audio_mixed", batch)  # torch.tensor
            self.assertIn("audio_ref", batch)  # torch.tensor

            self.assertIn("spectrogram_target", batch)  # torch.tensor
            self.assertIn("spectrogram_mixed", batch)  # torch.tensor
            self.assertIn("spectrogram_ref", batch)  # torch.tensor

    def test_dataloaders(self):
        _TOTAL_ITERATIONS = 10
        config_parser = ConfigParser.get_test_configs()
        with clear_log_folder_after_use(config_parser):
            dataloaders = get_dataloaders(config_parser, config_parser.get_text_encoder())
            for part in ["train", "val"]:
                dl = dataloaders[part]
                for i, batch in tqdm(enumerate(iter(dl)), total=_TOTAL_ITERATIONS,
                                     desc=f"Iterating over {part}"):
                    if i >= _TOTAL_ITERATIONS: break
