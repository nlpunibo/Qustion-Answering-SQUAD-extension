import random
import spacy
import json
import collections
import datasets
import transformers
from models.models import *
import numpy as np
from tqdm import tqdm, tqdm_notebook
from sklearn.metrics import accuracy_score


def fix_random(seed: int) -> None:
    """Fix all the possible sources of randomness.

    Args:
        seed: the seed to use.
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_text(answer: list) -> str:
    """Extract only the text from the answers.text column

    Args:
        answer: the answer.
    """
    return answer[0]

def get_json_data(json_path: str) -> dict:
    """Get the json data in form of a dictionary

    Args:
        json_path: path to the json file.
    """
    # Opening JSON file
    f = open(json_path)
    # returns JSON object as a dictionary
    json_data = json.load(f)
    # Closing file
    f.close()
    return json_data


class SQUAD():
    def __init__(self,
                 tokenizer, pad_on_right: bool = True, max_length: int = 384, doc_stride: int = 128,
                 squad_v2: bool = False, extension_tokenizer = None, device: str = 'cuda' , n_best_size: int = 20, max_answer_length: int = 50) -> None:
        """Class which exposes useful functions to process the dataset

          Args:
            tokenizer: tokenize the inputs (including converting the tokens to their corresponding IDs in the pretrained vocabulary)
                       and put it in a format the model expects, as well as generate the other inputs that model requires.
            pad_on_right: To work with any kind of models, we need to account for the special case
                          where the model expects padding on the left ( in which case we switch the order of the question and the context)
            max_length: The maximum length of a feature (question and context).
            doc_stride: The authorized overlap between two part of the context when splitting it is needed.
            squad_v2: it indicates if impossible answers are allowed or not.
            extension_tokenizer: tokenizer of one of the extension models
            device: cpu or cuda.
            n_best_size: Hyper-parameter to limit the number of possible answers
            max_answer_length: The maximum length of a predicted answer.
        """

        self.tokenizer = tokenizer
        self.pad_on_right = pad_on_right
        self.max_length = max_length
        self.doc_stride = doc_stride
        self.squad_v2 = squad_v2
        self.extension_tokenizer = extension_tokenizer
        self.device = device
        self.n_best_size = n_best_size
        self.max_answer_length = max_answer_length

    def prepare_train_features(self,
                               examples: collections.OrderedDict or dict) -> transformers.tokenization_utils_base.BatchEncoding:
        """Tokenize our examples with truncation and padding, but keep the overflows using a stride. This results
           in one example possible giving several features when a context is long, each of those features having a
           context that overlaps a bit the context of the previous feature.

          Args:
            examples: Squad samples
        """
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples.pop("offset_mapping")

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # CLS index
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # Start/end character index of the answer in the text.
            start_char = answers["answer_start"][0]
            end_char = start_char + len(answers["text"][0])

            # Start token index of the current span in the text.
            token_start_index = 0
            while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                token_start_index += 1

            # End token index of the current span in the text.
            token_end_index = len(input_ids) - 1
            while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                token_end_index -= 1

            # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
            if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                # Note: we could go after the last offset if the answer is the last word (edge case).
                while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                    token_start_index += 1
                tokenized_examples["start_positions"].append(token_start_index - 1)
                while offsets[token_end_index][1] >= end_char:
                    token_end_index -= 1
                tokenized_examples["end_positions"].append(token_end_index + 1)

        return tokenized_examples

    def prepare_train_features_classifier(self, example: dict) -> dict:
        """Tokenize our examples with truncation and padding.
          Args:
            examples: Dataset_classifier samples
        """

        context = example['context']
        question = example['question']
        answer = example['answers_text']

        # Context plus question
        question_plus = [x + " [SEP]" + y for x, y in zip(example['context'], example['question'])]

        encoder_inputs = self.tokenizer(question_plus, str(example['answers_text']),
                                   return_tensors='pt',
                                   max_length=self.max_length,
                                   truncation=True,
                                   padding="max_length")

        input_ids = encoder_inputs['input_ids'][0]
        input_attention = encoder_inputs['attention_mask'][0]

        outputs = {'input_ids': input_ids, 'attention_mask': input_attention,
                   'labels': example['label']}

        return outputs

    def encode_classifier(self, example: collections.OrderedDict or dict, answer: str) -> dict:

        context = example['context']
        question = example['question']

        question_plus = f" {str(question)}"
        question_plus += f"[SEP] {str(context)}"
        question_plus += f"[SEP] {str(answer)}"

        encoder_inputs = self.extension_tokenizer(question_plus, truncation=True, return_tensors='pt',
                                              max_length=self.max_length, pad_to_max_length=True)
        encoder_inputs = {k: v.to(self.device) for k, v in encoder_inputs.items()}

        return encoder_inputs

    def prepare_train_features_multiple_choice(self, example: dict) -> dict:
        """Tokenize our examples.

          Args:
            examples: Dataset_multiple_choice samples
        """

        context = example['context']
        question = example['question']
        answer1 = example['answer1']
        answer2 = example['answer2']
        answer3 = example['answer3']
        answer4 = example['answer4']
        answer5 = example['answer5']

        question_plus = f"{str(question)}"
        question_plus += f"[SEP] {str(context)}"

        encoder_inputs = self.tokenizer([[question_plus, str(answer1)],
                                    [question_plus, str(answer2)],
                                    [question_plus, str(answer3)],
                                    [question_plus, str(answer4)],
                                    [question_plus, str(answer5)]],
                                   return_tensors='pt',
                                   max_length=self.max_length,
                                   truncation=True,
                                   padding="max_length")

        input_ids = encoder_inputs['input_ids']
        input_attention = encoder_inputs['attention_mask']

        outputs = {'input_ids': input_ids, 'attention_mask': input_attention,
                   'labels': example['label']}
        return outputs

    def encode_multiplechoice(self, example: dict, answer_list: list) -> dict:

        context = example['context']
        question = example['question']
        answer1 = answer_list[0]
        answer2 = answer_list[1]
        answer3 = answer_list[2]
        answer4 = answer_list[3]
        answer5 = answer_list[4]

        question_plus = f"{str(question)}"
        question_plus += f"[SEP] {str(context)}"

        encoder_inputs = self.extension_tokenizer([[question_plus, str(answer1)],
                                                    [question_plus, str(answer2)],
                                                    [question_plus, str(answer3)],
                                                    [question_plus, str(answer4)],
                                                    [question_plus, str(answer5)]],
                                                   return_tensors='pt',
                                                   max_length=250,
                                                   truncation=True,
                                                   padding="max_length")

        encoder_inputs = {k: v.unsqueeze(0).to(self.device) for k, v in encoder_inputs.items()}

        return encoder_inputs

    def prepare_validation_features(self,
                                    examples: collections.OrderedDict or dict) -> transformers.tokenization_utils_base.BatchEncoding:
        """To check a given span is inside the context (and not the question) and to get back the text inside.
           To do this, we need to add two things to our validation features:
            - the ID of the example that generated the feature (since each example can generate several features, as seen before);
            - the offset mapping that will give us a map from token indices to character positions in the context.
           That's why we will re-process the validation set with the following function, slightly different from `prepare_train_features`

          Args:
            examples: Squad samples
        """
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        tokenized_examples = self.tokenizer(
            examples["question" if self.pad_on_right else "context"],
            examples["context" if self.pad_on_right else "question"],
            truncation="only_second" if self.pad_on_right else "only_first",
            max_length=self.max_length,
            stride=self.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")

        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        offset_mapping = tokenized_examples["offset_mapping"]

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []

        for i, offsets in enumerate(offset_mapping):
            # CLS index
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(self.tokenizer.cls_token_id)

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples["answers"][sample_index]

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if self.pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if self.pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

        # We keep the example_id that gave us this feature and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if self.pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        return tokenized_examples

    def postprocess_qa_predictions(self, examples: datasets.arrow_dataset.Dataset,
                                   features: datasets.arrow_dataset.Dataset, raw_predictions: tuple) -> collections.OrderedDict:
        """Function used to select the best answer from the raw predictions

          Args:
            examples: Squad samples
            features: Squad features
            raw_predictions: model predictions
        """

        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -self.n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -self.n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]
                        valid_answers.append(
                            {
                                "score": start_logits[start_index] + end_logits[end_index],
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer
            predictions[example["id"]] = best_answer["text"]

        return predictions

    def postprocess_qa_predictions_ner(self, examples: datasets.arrow_dataset.Dataset,
                                   features: datasets.arrow_dataset.Dataset,
                                       raw_predictions: tuple, ner_weights: dict) -> collections.OrderedDict:
        """Function used to select the best answer from the raw predictions

          Args:
            examples: Squad samples
            features: Squad features
            raw_predictions: model predictions
        """

        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Load the spacy ner model
        nlp = spacy.load('en_core_web_sm')

        # Most common types of questions
        question_heads = ['what', 'where', 'who', 'how', 'when', 'why', 'which']

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]

            # Extract the type of question
            question_head = example['question'].split()[0].lower()
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -self.n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -self.n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]

                        if question_head in question_heads:
                            ents = nlp(context[start_char: end_char]).ents
                            min = start_logits.min() + end_logits.min()
                            max = start_logits.max() + end_logits.max()
                            if not ents:
                                score = ner_weights[question_head]['OTHER'] + (
                                            (start_logits[start_index] + end_logits[end_index]) - min) / (max - min)
                            else:
                                score = 1
                                for ent in ents:
                                    score *= ner_weights[question_head][ent.label_]
                                score = score + ((start_logits[start_index] + end_logits[end_index]) - min) / (
                                            max - min)
                        else:
                            score = start_logits[start_index] + end_logits[end_index]


                        valid_answers.append(
                            {
                                "score": score,
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions

    def postprocess_qa_predictions_classifier(self, examples: datasets.arrow_dataset.Dataset,
                                   features: datasets.arrow_dataset.Dataset,
                                              raw_predictions: tuple, classifier: DistilBertClassifier) -> collections.OrderedDict:
        """Function used to select the best answer from the raw predictions

          Args:
            examples: Squad samples
            features: Squad features
            raw_predictions: model predictions
        """

        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]
                min_logit = start_logits.min() + end_logits.min()
                max_logit = start_logits.max() + end_logits.max()
                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -self.n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -self.n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]

                        score = start_logits[start_index] + end_logits[end_index]

                        valid_answers.append(
                            {
                                "score":  ((score-min_logit)/(max_logit-min_logit)),
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0:5]
                logit_list = []
                for answer in best_answer:
                    classifier_input = self.encode_classifier(example, answer['text'])
                    logit = classifier(**classifier_input)
                    logit = float(logit['logits'])
                    logit_list.append(logit)
                best_index = logit_list.index(max(logit_list))
                best_answer = best_answer[best_index]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions


    def postprocess_qa_predictions_multiplechoice(self, examples: datasets.arrow_dataset.Dataset,
                                   features: datasets.arrow_dataset.Dataset,
                                   raw_predictions: tuple,multiple_choice: DistilBertForMultipleChoice) -> collections.OrderedDict:
        """Function used to select the best answer from the raw predictions

          Args:
            examples: Squad samples
            features: Squad features
            raw_predictions: model predictions
        """
        all_start_logits, all_end_logits = raw_predictions
        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        features_per_example = collections.defaultdict(list)
        for i, feature in enumerate(features):
            features_per_example[example_id_to_index[feature["example_id"]]].append(i)

        # The dictionaries we have to fill.
        predictions = collections.OrderedDict()

        # Let's loop over all the examples!
        for example_index, example in enumerate(tqdm(examples)):
            # Those are the indices of the features associated to the current example.
            feature_indices = features_per_example[example_index]
            min_null_score = None  # Only used if squad_v2 is True.
            valid_answers = []

            context = example["context"]
            # Looping through all the features associated to the current example.
            for feature_index in feature_indices:
                # We grab the predictions of the model for this feature.
                start_logits = all_start_logits[feature_index]
                end_logits = all_end_logits[feature_index]

                # min = start_logits.min() + end_logits.min()
                # max = start_logits.max() + end_logits.max()

                # This is what will allow us to map some the positions in our logits to span of texts in the original
                # context.
                offset_mapping = features[feature_index]["offset_mapping"]

                # Update minimum null prediction.
                cls_index = features[feature_index]["input_ids"].index(self.tokenizer.cls_token_id)
                feature_null_score = start_logits[cls_index] + end_logits[cls_index]
                if min_null_score is None or min_null_score < feature_null_score:
                    min_null_score = feature_null_score

                # Go through all possibilities for the `n_best_size` greater start and end logits.
                start_indexes = np.argsort(start_logits)[-1: -self.n_best_size - 1: -1].tolist()
                end_indexes = np.argsort(end_logits)[-1: -self.n_best_size - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                                start_index >= len(offset_mapping)
                                or end_index >= len(offset_mapping)
                                or offset_mapping[start_index] is None
                                or offset_mapping[end_index] is None
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self.max_answer_length:
                            continue

                        start_char = offset_mapping[start_index][0]
                        end_char = offset_mapping[end_index][1]

                        score = start_logits[start_index] + end_logits[end_index]

                        valid_answers.append(
                            {
                                "score": score,
                                "text": context[start_char: end_char]
                            }
                        )

            if len(valid_answers) > 0:
                best_answer = sorted(valid_answers, key=lambda x: x["score"], reverse=True)[0:5]
                best_answer_text = [a_dict['text'] for a_dict in best_answer]
                best_answer_text = best_answer_text + ['[PAD]'] * (5 - len(best_answer_text))
                multiple_choice_input = self.encode_multiplechoice(example, best_answer_text)
                logits = multiple_choice(**multiple_choice_input)
                min = logits['logits'][0].min()
                max = logits['logits'][0].max()

                for i, answer in enumerate(best_answer):
                    answer['score'] = 0.9 * answer['score'] + 0.2 * float(logits['logits'][0][i])
                    # answer['score'] += 0.07*(float(logits['logits'][0][i])-min)/(max-min)

                best_answer = sorted(best_answer, key=lambda x: x["score"], reverse=True)[0]
            else:
                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                best_answer = {"text": "", "score": 0.0}

            # Let's pick our final answer: the best one or the null answer (only for squad_v2)
            if not self.squad_v2:
                predictions[example["id"]] = best_answer["text"]
            else:
                answer = best_answer["text"] if best_answer["score"] > min_null_score else ""
                predictions[example["id"]] = answer

        return predictions


class Eval():
    def __init__(self, validation_features: datasets.arrow_dataset.Dataset, val_data: datasets.dataset_dict.DatasetDict,
                 metric, squad: SQUAD) -> None:
        """Class used to do the evaluation, during training, on the validation set

          Args:
            validation_features: The Validation feature dataset
            val_data: The Validation dataset
            metric: The metric(from the datasets library) used to evaluate the results
            squad: An instance of the SQUAD class
        """

        self.validation_features = validation_features
        self.val_data = val_data
        self.metric = metric
        self.squad = squad

    def compute_metrics(self, pred: transformers.trainer_utils.EvalPrediction) -> dict:
        # The Trainer hides the columns that are not used by the model
        # (here example_id and offset_mapping which we will need for our post-processing), so we set them back
        self.validation_features.set_format(type=self.validation_features.format["type"],
                                            columns=list(self.validation_features.features.keys()))

        # To get the final predictions we can apply our post-processing function to our raw predictions
        final_predictions = self.squad.postprocess_qa_predictions(self.val_data["train"], self.validation_features,
                                                                  pred.predictions)

        # We just need to format predictions and labels a bit as metric expects a list of dictionaries and not one big dictionary
        formatted_predictions = [{"id": k, "prediction_text": v} for k, v in final_predictions.items()]
        references = [{"id": ex["id"], "answers": ex["answers"]} for ex in self.val_data["train"]]

        # Hide again the columns that are not used by the model
        self.validation_features.set_format(type=self.validation_features.format["type"],
                                            columns=['attention_mask', 'end_positions', 'input_ids', 'start_positions'])
        metrics = self.metric.compute(predictions=formatted_predictions, references=references)

        return metrics


def compute_metrics_classifier(pred):
    labels = pred.label_ids

    preds = [np.round(num) for num in pred.predictions]

    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def compute_metrics_multiple_choice(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc
    }