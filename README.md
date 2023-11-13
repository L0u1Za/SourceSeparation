# ASR project

## Report

You may see the report if you follow [this link](https://api.wandb.ai/links/l0u1za/kvm4x41t)

## Installation guide

Firstly, install needed requirements for running model

```shell
pip install -r ./requirements.txt
```

### Download model

Use bash script to download trained model

```shell
cd ./default_test_model
./download.sh
```

It will be placed to `./default_test_model/checkpoint.pth`

If you have some issues using bash utilities, you may download model directly from [google drive](https://drive.google.com/file/d/1lr14jvV3M3zm75KoLrJcMnsFrQ4OYtp1/view?usp=sharing)

## Mix datasets

To mix datasets, use `mix.py` file
```shell
python mix.py -p <path_to_dataset> -o <path_to_output_mixed_dataset>
```
Or just add needed `MixedDataset` to config file, it will automatically download LibriSpeech to create mixes.

## Run unit tests

You may check the correct work of implementation using unit tests

```shell
python -m unittest discover hw_ss/tests
```

## Run test model with prepared configuration

```shell
python test.py \
   -r default_test_model/checkpoint.pth \
   -t test_data \
   -o test_result.json
```

## Credits

This repository is based on a heavily modified fork
of [pytorch-template](https://github.com/victoresque/pytorch-template) repository.

## Docker

You can use this project with docker. Quick start:

```bash
docker build -t my_hw_ss_image .
docker run \
   --gpus '"device=0"' \
   -it --rm \
   -v /path/to/local/storage/dir:/repos/asr_project_template/data/datasets \
   -e WANDB_API_KEY=<your_wandb_api_key> \
	my_hw_ss_image python -m unittest
```

Notes:

* `-v /out/of/container/path:/inside/container/path` -- bind mount a path, so you wouldn't have to download datasets at
  the start of every docker run.
* `-e WANDB_API_KEY=<your_wandb_api_key>` -- set envvar for wandb (if you want to use it). You can find your API key
  here: https://wandb.ai/authorize
