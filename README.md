# Qustion-Answering-Squad-extension

In this extension repository of our base project [Question Answering SQUAD](https://github.com/nlpunibo/Question-Answering-SQUAD) we present 4 diferent approaches that we have tried to improve the performances, in terms of F1 score and EM of our model.

## Installation

To quickly try out our experiments, clone this repository and install the necessary requirements by running

`pip install -r requirements.txt`

We recommend creating a separate python 3.6 environment. 

## Usage

To the script `compute_answers.py`, you just need to launch it:
- `python3 compute_answers.py --extension name_of_the_extension --test test_dataset --weights path_to_ner_weights`

Where:
- `--extension` with this parameter you can choose the extension you want to try. It can assume only 4 values [ner, multiple_choice, classifier, convolutional_classifier]
- `--test` this parameter is used to choose for which dataset you want to compute the predictions, so you can pass to it the path to the json file of your dataset, or "squad_v1" or "squad_v2" if you want to compute the predictions for respectively the squad 1.1 dev_set and the squad 2.0 dev_set.

Then if youy want to evaluate the results you can use the  `evaluate.py` script to test the method you have chosen:
- `python3 evaluate.py *path_to_ground_truth* *path_to_predictions_file*`

## Recommendations

We strongly reccomend you to use a GPU for running the `train.py` and the `compute_answers.py` scripts. To do so you can use the Nvidia graphic card of your machine, if it has one. In this case make sure that you have all the prerequisites (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) and to have installed the pytorch version for the CUDA platform (https://pytorch.org/).

## Authors

**Simone Gayed Said** - simone.gayed@studio.unibo.it </br>
**Alex Rossi** - alex.rossi6@studio.unibo.it </br>
**Jia Liang Zhou** - jialiang.zhou@studio.unibo.it </br>
**Hanying Zhang** - hanying.zhang@studio.unibo.it

## Useful Links

**Hugging Face library** - https://huggingface.co/transformers/ </br>
**Our organization on the Hugging Face Hub** - https://huggingface.co/nlpunibo
