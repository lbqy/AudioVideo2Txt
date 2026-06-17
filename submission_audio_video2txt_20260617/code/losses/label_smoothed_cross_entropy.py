import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
from omegaconf import II
from fairseq import metrics
from fairseq.dataclass import FairseqDataclass

import utils


@dataclass
class LabelSmoothedCrossEntropyCriterionConfig(FairseqDataclass):
    label_smoothing: float = field(
        default=0.0,
        metadata={"help": "epsilon for label smoothing, 0 means no label smoothing"},
    )
    focal_gamma: float = field(
        default=0.0,
        metadata={"help": "gamma for focal loss modulation, 0 disables focal loss"},
    )
    confidence_penalty: float = field(
        default=0.0,
        metadata={"help": "coefficient for confidence penalty regularization"},
    )
    ctc_weight: float = field(
        default=0.0,
        metadata={"help": "weight for auxiliary CTC loss, 0 disables CTC"},
    )
    report_accuracy: bool = field(
        default=False,
        metadata={"help": "report accuracy metric"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    loss = None
    nll_loss = None
    ################################################################################
    # TODO:                                                                        #
    #  finish the forward                                                          #
    #  return:                                                                     #
    #   - loss: the loss after smoothing                                           #
    #   - nll_loss: the standard nll loss                                          #
    #  NOTE: if `reduce=True`, sum the loss of all tokens and return a scalar      #
    ################################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # make target broadcastable against lprobs: [N] -> [N, 1]
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)

    # standard negative log-likelihood: pick the log-prob of the gold token
    nll_loss = -lprobs.gather(dim=-1, index=target)
    # label smoothing term: the uniform component spreads probability mass over
    # the whole vocabulary, i.e. -sum_i log p_i
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)

    if ignore_index is not None:
        # zero-out the loss contributed by padding/ignored positions
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)

    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()

    # combine: (1 - eps) on the gold token + eps spread uniformly over the
    # remaining (V - 1) classes
    eps_i = epsilon / (lprobs.size(-1) - 1)
    loss = (1.0 - epsilon - eps_i) * nll_loss + eps_i * smooth_loss

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ################################################################################
    #                              END OF YOUR CODE                                #
    ################################################################################
    return loss, nll_loss


class LabelSmoothedCrossEntropyCriterion(_Loss):
    def __init__(
        self,
        cfg,
        tgt_dict
    ):
        super().__init__()

        self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
        self.sentence_avg = cfg.sentence_avg
        self.eps = cfg.label_smoothing
        self.focal_gamma = getattr(cfg, "focal_gamma", 0.0)
        self.confidence_penalty = getattr(cfg, "confidence_penalty", 0.0)
        self.ctc_weight = getattr(cfg, "ctc_weight", 0.0)
        self.ignore_prefix_size = cfg.ignore_prefix_size
        self.report_accuracy = cfg.report_accuracy

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        ce_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        loss = ce_loss
        ctc_loss = None
        if self.ctc_weight > 0:
            ctc_loss = self.compute_ctc_loss(model, net_output, sample, reduce=reduce)
            loss = loss + self.ctc_weight * ctc_loss
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "total_loss": loss.data,
            "ce_loss": ce_loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if ctc_loss is not None:
            logging_output["ctc_loss"] = ctc_loss.data
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_decoder_output(self, net_output):
        return net_output["decoder_out"] if isinstance(net_output, dict) else net_output

    def get_lprobs_and_target(self, model, net_output, sample):
        net_output = self.get_decoder_output(net_output)
        if isinstance(net_output, tuple):
            net_output = (net_output[0].float(),) + net_output[1:]
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        return lprobs.float().view(-1, lprobs.size(-1)), target.view(-1)

    def compute_ctc_loss(self, model, net_output, sample, reduce=True):
        if not isinstance(net_output, dict) or "encoder_out" not in net_output:
            raise ValueError("criterion.ctc_weight > 0 requires model CTC output")
        if not hasattr(model, "ctc_proj") or model.ctc_proj is None:
            raise ValueError("criterion.ctc_weight > 0 requires model.ctc_weight > 0")

        encoder_out = net_output["encoder_out"]["encoder_out"]
        padding_mask = net_output["encoder_out"].get("padding_mask", None)
        ctc_logits = model.ctc_proj(encoder_out.to(model.ctc_proj.weight.dtype))
        ctc_lprobs = F.log_softmax(ctc_logits.float(), dim=-1)

        if padding_mask is None:
            input_lengths = torch.full(
                (encoder_out.size(1),),
                encoder_out.size(0),
                dtype=torch.long,
                device=encoder_out.device,
            )
        else:
            input_lengths = padding_mask.logical_not().long().sum(dim=1)

        target = sample["target"]
        target_mask = target.ne(self.padding_idx)
        if hasattr(model, "tgt_dict") and model.tgt_dict is not None:
            target_mask = target_mask & target.ne(model.tgt_dict.eos())
        target_lengths = target_mask.long().sum(dim=1)
        flat_target = target.masked_select(target_mask)

        return F.ctc_loss(
            ctc_lprobs,
            flat_target,
            input_lengths,
            target_lengths,
            blank=self.padding_idx,
            reduction="sum" if reduce else "none",
            zero_infinity=True,
        )

    def compute_enhanced_loss(self, lprobs, target, reduce=True):
        loss = None
        nll_loss = None
        ################################################################################
        # TODO:                                                                        #
        #  extend label-smoothed cross entropy with optional regularizers              #
        #  1. call label_smoothed_nll_loss with reduce=False to keep per-token losses  #
        #  2. build a non-padding mask from target and self.padding_idx                #
        #  3. if self.focal_gamma > 0, multiply each token loss by                    #
        #     (1 - p_t) ** self.focal_gamma, where p_t = exp(-nll_loss)                #
        #  4. if self.confidence_penalty > 0, add                                     #
        #     self.confidence_penalty * sum(p * log(p)) for each non-padding token     #
        #  5. if reduce=True, sum both loss and nll_loss before returning              #
        ################################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=False,
        )

        non_pad_mask = target.ne(self.padding_idx)
        if loss.dim() > non_pad_mask.dim():
            non_pad_mask = non_pad_mask.unsqueeze(-1)

        if self.focal_gamma > 0:
            p_t = torch.exp(-nll_loss)
            focal_factor = (1.0 - p_t) ** self.focal_gamma
            loss = loss * focal_factor

        if self.confidence_penalty > 0:
            confidence_loss = (lprobs.exp() * lprobs).sum(dim=-1, keepdim=True)
            loss = loss + self.confidence_penalty * confidence_loss * non_pad_mask

        if reduce:
            loss = loss.sum()
            nll_loss = nll_loss.sum()

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ################################################################################
        #                              END OF YOUR CODE                                #
        ################################################################################
        return loss, nll_loss

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        if self.focal_gamma > 0 or self.confidence_penalty > 0:
            loss, nll_loss = self.compute_enhanced_loss(lprobs, target, reduce=reduce)
        else:
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs,
                target,
                self.eps,
                ignore_index=self.padding_idx,
                reduce=reduce,
            )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        total_loss_sum = sum(log.get("total_loss", 0) for log in logging_outputs)
        ce_loss_sum = sum(log.get("ce_loss", 0) for log in logging_outputs)
        ctc_loss_sum = sum(log.get("ctc_loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        if total_loss_sum != 0:
            metrics.log_scalar(
                "total_loss",
                total_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ce_loss_sum != 0:
            metrics.log_scalar(
                "ce_loss",
                ce_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        if ctc_loss_sum != 0:
            metrics.log_scalar(
                "ctc_loss",
                ctc_loss_sum / sample_size / math.log(2),
                sample_size,
                round=3,
            )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
