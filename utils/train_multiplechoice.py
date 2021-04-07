import argparse
import pandas as pd
from utils.utils import *
from models.models import *
from pathlib import Path
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, TrainingArguments, Trainer, default_data_collator


def main():
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument("path_to_dataset_file", help="Path to the multiple choice dataset file", type=str)
    args = parser.parse_args()

    # Set the seed for reproducibility
    fix_random(seed=42)

    # Load the data
    data_path = Path(args.path_to_dataset_file).parent

    df = pd.read_csv(args.path_to_dataset_file, index_col=0)

    train, val = train_test_split(df, test_size=0.2)

    train.to_csv(str(data_path / "dataset_multiple_choice_train.csv"), encoding='utf-8')
    val.to_csv(str(data_path / "dataset_multiple_choice_val.csv"), encoding='utf-8')

    train_data = load_dataset('csv', data_files=str(data_path / "dataset_multiple_choice_train.csv"))
    val_data = load_dataset('csv', data_files=str(data_path / "dataset_multiple_choice_val.csv"))

    # Preprocessing the training data

    # Before we can feed those texts to our model, we need to preprocess them.
    # This is done by a Transformers Tokenizer which will(as the name indicates) tokenize
    # the inputs(including converting the tokens to their corresponding IDs in the pretrained
    # vocabulary) and put it in a format the model expects, as well as generate the
    # other inputs that model requires.

    # To do all of this, we instantiate our tokenizer
    # with the AutoTokenizer.from_pretrained method, which will ensure:
    # - we get a tokenizer that corresponds to the model architecture we
    #   want to use,
    # - we download the vocabulary used when pretraining this specific
    #   checkpoint.
    model_checkpoint = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # The following assertion ensures that our tokenizer is a fast tokenizers (backed by Rust) from the HugginFace Tokenizers library.
    # Those fast tokenizers are available for almost all models, and we will need some of the special features they have for our preprocessing.
    assert isinstance(tokenizer, transformers.PreTrainedTokenizerFast)

    # The maximum length of a feature (question and context)
    max_length = 384

    # The authorized overlap between two part of the context when splitting it is needed.
    doc_stride = 128

    # To work with any kind of models, we need to account for the special case
    # where the model expects padding on the left ( in which case we switch the order of the question and the context)
    pad_on_right = tokenizer.padding_side == "right"

    # To apply the TokenizeData.prepare_train_features function on all the sentences (or pairs of sentences) in our dataset,
    # we just use the map method of our dataset object we created earlier. This will apply the function on all the elements
    # of all the splits in dataset, so our training, validation and testing data will be preprocessed in one single command.
    # Since our preprocessing changes the number of samples, we need to remove the old columns when applying it.

    # Note that we passed batched=True to encode the texts by batches together.
    # This is to leverage the full benefit of the fast tokenizer we loaded earlier,
    # which will use multi-threading to treat the texts in a batch concurrently.
    squad = SQUAD(tokenizer, pad_on_right, max_length, doc_stride)
    train_tokenized_datasets = train_data.map(squad.prepare_train_features_multiple_choice,
                                              remove_columns=train_data['train'].column_names)
    val_tokenized_datasets = val_data.map(squad.prepare_train_features_multiple_choice, remove_columns=val_data['train'].column_names)

    # Now that our data is ready for training, we can download the pretrained model and fine-tune it.
    model = DistilBertForMultipleChoice.from_pretrained(model_checkpoint)

    # Tell pytorch to run this model on the GPU.
    if torch.cuda.is_available():
        model.cuda()

    # The warning is telling us we are throwing away some weights (the `vocab_transform` and `vocab_layer_norm` layers)
    # and randomly initializing some other (the `pre_classifier` and `classifier` layers).
    # This is absolutely normal in this case, because we are removing the head used to pretrain the model
    # on a masked language modeling objective and replacing it with a new head for which we don't have pretrained weights,
    # so the library warns us we should fine-tune this model before using it for inference, which is exactly what we are going to do.

    # To instantiate a `Trainer`, we will need to define three more things.
    # The most important is the `TrainingArguments`, which is a class that contains all the attributes to customize the training.
    # It requires one folder name, which will be used to save the checkpoints of the model, and all other arguments are optional

    # Here we set the evaluation to be done at the end of each epoch, tweak the learning rate,
    # use the defined `batch_size` and customize the number of epochs for training, as well as the weight decay.
    batch_size = 16
    args = TrainingArguments(
        output_dir='../src/results',
        save_total_limit=1,
        evaluation_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir='./logs',
        label_names=["labels"]
    )

    # Then we will need a data collator that will batch our processed examples together, here the default one will work.
    data_collator = default_data_collator

    # Then we just need to pass all of this along with our datasets to the Trainer
    trainer = Trainer(
        model,
        args,
        compute_metrics=compute_metrics_multiple_choice,
        train_dataset=train_tokenized_datasets["train"],
        eval_dataset=val_tokenized_datasets["train"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # Now we can now finetune our model by just calling the train method
    trainer.train()

    # Since this training is particularly long, let's save the model just in case we need to restart.
    trainer.save_model(str(data_path / "multiple_choice_model"))


if __name__ == '__main__':
    main()
