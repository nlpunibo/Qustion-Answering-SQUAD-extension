import math
import torch

from transformers.modeling_outputs import QuestionAnsweringModelOutput
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_outputs import MultipleChoiceModelOutput
from transformers import DistilBertPreTrainedModel
from transformers import DistilBertModel

from torch import nn
from torch.nn import Conv1d


def gelu(x):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in
    torch.nn.functional Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class DistilBertForQuestionAnswering(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        self.qa_outputs_0 = nn.Linear(config.dim, 512)
        self.qa_outputs_1 = nn.Linear(512, 32)
        self.qa_outputs = nn.Linear(32, config.num_labels)

        assert config.num_labels == 2
        self.dropout = nn.Dropout(config.qa_dropout)

        self.LayerNorm = nn.LayerNorm(normalized_shape=[384, 2])

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = distilbert_output[0]  # (bs, max_query_len, dim)
        hidden_states = self.dropout(hidden_states)  # (bs, max_query_len, dim)

        logits = gelu_new(self.qa_outputs_0(hidden_states))  # (bs, max_query_len, 2)
        logits = gelu_new(self.qa_outputs_1(logits))
        # logits = self.LayerNorm_0(logits)

        logits = self.qa_outputs(logits)
        logits = self.LayerNorm(logits)

        start_logits, end_logits = logits.split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)  # (bs, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (bs, max_query_len)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions
        )


class DistilBertClassifier(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, self.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_state = distilbert_output[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = nn.ReLU()(pooled_output)  # (bs, dim)
        # pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, num_labels)
        # logits = logits.to(torch.float32)
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()

            loss = loss_fct(logits.view(-1), labels.to(torch.float32).view(-1))

        if not return_dict:
            output = (logits,) + distilbert_output[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions,
        )


class DistilBertForMultipleChoice(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)
        self.pre_classifier = nn.Linear(768, 512)
        self.pre_classifier2 = nn.Linear(512, 256)
        self.pre_classifier3 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 1)
        self.dropout = nn.Dropout(config.seq_classif_dropout)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the multiple choice classification loss. Indices should be in ``[0, ...,
            num_choices-1]`` where :obj:`num_choices` is the size of the second dimension of the input tensors. (See
            :obj:`input_ids` above)

        Returns:

        Examples::

            >>> from transformers import DistilBertTokenizer, DistilBertForMultipleChoice
            >>> import torch

            >>> tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
            >>> model = DistilBertForMultipleChoice.from_pretrained('distilbert-base-cased')

            >>> prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
            >>> choice0 = "It is eaten with a fork and a knife."
            >>> choice1 = "It is eaten while held in the hand."
            >>> labels = torch.tensor(0).unsqueeze(0)  # choice0 is correct (according to Wikipedia ;)), batch size 1

            >>> encoding = tokenizer([[prompt, choice0], [prompt, choice1]], return_tensors='pt', padding=True)
            >>> outputs = model(**{k: v.unsqueeze(0) for k,v in encoding.items()}, labels=labels) # batch size is 1

            >>> # the linear classifier still needs to be trained
            >>> loss = outputs.loss
            >>> logits = outputs.logits
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)

        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        pooled_output = self.pre_classifier2(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        pooled_output = self.pre_classifier3(pooled_output)
        pooled_output = nn.ReLU()(pooled_output)
        pooled_output = self.dropout(pooled_output)

        logits = self.classifier(pooled_output)  # (bs * num_choices, 1)

        reshaped_logits = logits.view(-1, num_choices)  # (bs, num_choices)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class DistilBertConvolutionalClassifier(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.distilbert = DistilBertModel(config)

        '''
          Adding a fully-convolutional classifier inspired by DeepLab V3+ used for 
          semantic segmentation.
          Using a Atrous Spatial Pyramid Pooling (ASPP) module with dilated convolutions at 
          different dilation rates, in order to capture larger coarsear structures,
          possibly capturing better the structure of an answer.
          The ASPP modules halves the size of the input sequence, so we upsample x2 after
          the scoring layer.
        '''
        # Spatial Pyramid Pooling with dilated convs
        self.qa_aspp_r3 = Conv1d(config.dim, config.dim, 3, stride=2, dilation=6, groups=config.dim, padding=6)
        self.qa_aspp_r3_1x1 = Conv1d(config.dim, config.dim // 4, 1)

        self.qa_aspp_r6 = Conv1d(config.dim, config.dim, 3, stride=2, dilation=12, groups=config.dim, padding=12)
        self.qa_aspp_r6_1x1 = Conv1d(config.dim, config.dim // 4, 1)

        self.qa_aspp_r12 = Conv1d(config.dim, config.dim, 3, stride=2, dilation=18, groups=config.dim, padding=18)
        self.qa_aspp_r12_1x1 = Conv1d(config.dim, config.dim // 4, 1)

        self.qa_aspp_score = Conv1d(config.dim // 4 * 3, config.num_labels, 1)
        # self.LayerNorm_aspp = nn.LayerNorm(normalized_shape = [384,2])
        self.upsampling2D = nn.Upsample(scale_factor=2, mode='bilinear')

        assert config.num_labels == 2
        # self.dropout = nn.Dropout(config.qa_dropout)

        # self.LayerNorm = nn.LayerNorm(normalized_shape = [384,2])

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            head_mask=None,
            inputs_embeds=None,
            start_positions=None,
            end_positions=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        """
        start_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        end_positions (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (:obj:`sequence_length`). Position outside of the
            sequence are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        distilbert_output = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = distilbert_output[0].permute(0, 2, 1)

        aspp_r3 = gelu_new(self.qa_aspp_r6_1x1(self.qa_aspp_r3(logits)))
        aspp_r6 = gelu_new(self.qa_aspp_r6_1x1(self.qa_aspp_r6(logits)))
        aspp_r12 = gelu_new(self.qa_aspp_r12_1x1(self.qa_aspp_r12(logits)))

        out_aspp = torch.cat((aspp_r3, aspp_r6, aspp_r12), 1)

        logits = gelu_new(self.qa_aspp_score(out_aspp))

        logits = logits.unsqueeze(dim=3)

        logits = self.upsampling2D(logits)
        logits = logits[:, :, :, 0]

        start_logits, end_logits = logits.permute(0, 2, 1).split(1, dim=-1)

        start_logits = start_logits.squeeze(-1)  # (batch_size, max_query_len)
        end_logits = end_logits.squeeze(-1)  # (batch_size, max_query_len)

        # print(start_logits.shape)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + distilbert_output[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=distilbert_output.hidden_states,
            attentions=distilbert_output.attentions
        )
