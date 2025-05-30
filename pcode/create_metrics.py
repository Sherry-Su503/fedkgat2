# -*- coding: utf-8 -*-
import math
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


class Metrics(object):
    """根据不同的任务类型，选择合适的评估指标并计算。它可以支持多种任务类型：
    分类任务：计算 top-k 准确率。
语言模型任务：计算困惑度。
神经机器翻译任务：计算困惑度和 top-1 准确率。
推荐系统任务：计算 AUC 和 F1 分数"""

    def __init__(self, model, task="classification"):
        self.model = model
        self.task = task
        self.metric_names = None
        self.metrics_fn = self._infer()

    def evaluate(self, loss, output, target, **kwargs):
        """根据任务类型调用相应的评估函数，返回相应的评估指标结果"""
        return self.metrics_fn(loss, output, target, **kwargs)

    def _infer(self):
        if self.task == "classification":
            self.topks = (
                (1, 5)
                if getattr(self.model, "num_classes", None) is not None
                   and self.model.num_classes >= 5
                else (1,)
            )
            self.metric_names = ["top{}".format(topk) for topk in self.topks]
            return self._accuracy
        elif self.task == "language_modeling":
            self.metric_names = ["ppl"]
            return self._ppl
        elif self.task == "transformer_nmt":
            self.metric_names = ["ppl", "top1"]
            return self._transformer_nmt
        elif self.task == "recommondation":
            self.metric_names = ["auc", "f1"]
            return self._recommondation
        elif self.task == "null":
            self.metric_names = ["null"]
            return self._null
        else:
            raise NotImplementedError

        # some safety check.
        assert self.metric_names is not None

    def _accuracy(self, loss, output, target):
        """Computes the precision@k for the specified values of k"""
        res = []

        if len(self.topks) > 0:
            maxk = max(self.topks)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.reshape(1, -1).expand_as(pred))

            for topk in self.topks:
                correct_k = correct[:topk].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size).item())
        else:
            res += [0]
        return res

    def _ppl(self, loss, output, target):
        return [math.exp(loss)]

    def _transformer_nmt(self, loss, output, target, **kwargs):
        pred = output.max(1)[1]
        n_correct = pred.eq(target)
        n_correct = n_correct.masked_select(kwargs["non_pad_mask"]).sum().item()
        return [math.exp(loss), n_correct / kwargs["n_samples"]]

    def _recommondation(self, loss, output, target):
        if len(target.unique()) != 1:
            target = target.cpu().detach().numpy()
            output = output.cpu().detach().numpy()
            auc = roc_auc_score(target, output)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            f1 = f1_score(target, output)
        else:
            auc = f1 = 0
        return [auc, f1]

    def _null(self, loss, output, target):
        return [0];
