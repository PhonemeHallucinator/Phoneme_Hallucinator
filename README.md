# Phoneme_Hallucinator
This is the repository of paper "Phoneme Hallucinator: One-shot Voice Conversion via Set Expansion" under double-blind review.

## Training Tutorial
1. To prepare the training set, we need to use WavLM to extract speech representations. Go to [kNN-VC repo](https://github.com/bshall/knn-vc) and follow its instructions to extract speech representations. Namely, after placing LibriSpeech dataset in a correct location, run the command:

   `python prematch_dataset.py --librispeech_path /path/to/librispeech/root --out_path /path/where/you/want/outputs/to/go --topk 4 --matching_layer 6 --synthesis_layer 6`

   Note that we don't use the "--prematch" option, becuase we only need to extract representations, not to extract and then perform kNN regression.
