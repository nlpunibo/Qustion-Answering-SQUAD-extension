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
- `extension` is the name of the extension you want to try and it can assume only 4 values [ner, multiple_choice, classifier, convolutional_classifier]
- `test` this parameter is used to choose for which dataset you want to compute the predictions, so you can pass to it the path to the json file of your dataset, or "squad_v1" or "squad_v2" if you want to compute the predictions for respectively the squad 1.1 dev_set and the squad 2.0 dev_set

```json
{
    "question_id": "textual answer",
    ...
}
```
 - `python3 evaluate.py path_to_json_file prediction_file`: given the path to the same testing json file used in the `compute_answers.py` script and the json file produced by the script itself, prints to the standard output a dictionary of metrics such as the `F1` and `Exact Match` scores, which can be used to assess the performance of a trained model as done in the official SQuAD competition

The two Colab notebooks `DistilbertQA_train.ipynb` and `DistilbertQA_eval.ipynb` provide more comments and useful plots w.r.t the python scripts. If you want to use them make sure to have a Google Drive folder with the json files you want to use and to change in the notebooks the `FOLDER_NAME` and `JSON_TEST_FILE` text fields.

## Recommendations

We strongly reccomend you to use a GPU for running the `train.py` and the `compute_answers.py` scripts. To do so you can use the Nvidia graphic card of your machine, if it has one. In this case make sure that you have all the prerequisites (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html#install-windows) and to have installed the pytorch version for the CUDA platform (https://pytorch.org/).

If you don't have an Nvidia GPU at your disposal don't worry we have created for you two Colab Notebooks `DistilbertQA_train.ipynb` and the `DistilbertQA_eval.ipynb`. Colab is a hosted Jupyter notebook service that requires no setup to use, while providing free access to computing resources including GPUs! You will not have to install anything, just navigate to Edit→Notebook Settings, and make sure that GPU is selected as Hardware Accelerator.

## Authors

**Simone Gayed Said** - simone.gayed@studio.unibo.it </br>
**Alex Rossi** - alex.rossi6@studio.unibo.it </br>
**Jia Liang Zhou** - jialiang.zhou@studio.unibo.it </br>
**Hanying Zhang** - hanying.zhang@studio.unibo.it

## Useful Links

**Hugging Face library** - https://huggingface.co/transformers/ </br>
**Our organization on the Hugging Face Hub** - https://huggingface.co/nlpunibo
