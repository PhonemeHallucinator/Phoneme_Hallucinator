# Phoneme_Hallucinator
This is the repository of paper "Phoneme Hallucinator: One-shot Voice Conversion via Set Expansion" under double-blind review.

## Training Tutorial
1. To prepare the training set, we need to use WavLM to extract speech representations. Go to [kNN-VC repo](https://github.com/bshall/knn-vc) and follow its instructions to extract speech representations. Namely, after placing LibriSpeech dataset in a correct location, run the command:

   `python prematch_dataset.py --librispeech_path /path/to/librispeech/root --out_path /path/where/you/want/outputs/to/go --topk 4 --matching_layer 6 --synthesis_layer 6`

   Note that we don't use the "--prematch" option, becuase we only need to extract representations, not to extract and then perform kNN regression.

2. After the above step, you can get a `--out_path` folder with three subfolders `train-clean-100`, `test-clean` and `dev-clean` where each folder contains the speech representation files (".pt").
3. Go to our repo `./dataset/speech.py` and change the variables `path_to_wavlm_feat` and `tfrecord_path` accordingly. You need to change `path_to_wavlm_feat` to where the speech representations are stored in the previous step. If `tfrecord_path` doesn't exist, our codes will create tfrecords and save them to `tfrecord_path`.

