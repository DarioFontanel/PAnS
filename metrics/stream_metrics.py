import numpy as np
import torch
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics as sk

recall_level_default = 0.95


class _StreamMetrics(object):
    def __init__(self):
        """ Overridden by subclasses """
        pass

    def update(self, gt, pred):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def get_results(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def to_str(self, metrics):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def reset(self):
        """ Overridden by subclasses """
        raise NotImplementedError()

    def synch(self, device):
        """ Overridden by subclasses """
        raise NotImplementedError()


class StreamSegMetrics(_StreamMetrics):
    """
    Stream Metrics for Semantic Segmentation Task
    """

    def __init__(self, n_classes, unk_class=13):
        super().__init__()
        self.n_classes = n_classes
        self.unk_class = unk_class
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        self.total_samples = 0
        self.auroc = 0
        self.aupr = 0
        self.fpr = 0
        self.total_test_images = 0

    def update(self, label_trues, label_preds=None, prob_preds=None):
        if label_preds is not None:
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())
            self.total_samples += len(label_trues)

        out_label = label_trues >= self.unk_class
        in_scores = 1 - prob_preds[np.logical_not(out_label)]
        out_scores = 1 - prob_preds[out_label]

        if (len(out_scores) != 0) and (len(in_scores) != 0):
            auroc, aupr, fpr = self.compute_as_metrics(out_scores, in_scores)
            self.auroc += auroc
            self.aupr += aupr
            self.fpr += fpr
            self.total_test_images += 1
        else:
            print("This image does not contain any OOD pixels or is only OOD.")

    def compute_as_metrics(self, _pos, _neg, recall_level=recall_level_default):
        pos = np.array(_pos[:]).reshape((-1, 1))
        neg = np.array(_neg[:]).reshape((-1, 1))
        examples = np.squeeze(np.vstack((pos, neg)))
        labels = np.zeros(len(examples), dtype=np.int32)
        labels[:len(pos)] += 1

        auroc = sk.roc_auc_score(labels, examples)
        aupr = sk.average_precision_score(labels, examples)
        fpr = self.fpr_and_fdr_at_recall(labels, examples, recall_level)

        return auroc, aupr, fpr

    def fpr_and_fdr_at_recall(self, y_true, y_score, recall_level=recall_level_default, pos_label=None):
        classes = np.unique(y_true)
        if (pos_label is None and
                not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # True per ogni pixel appartenente a unknown, false per ogni pixel appartenente a known (una sorta
        # di nuovo vettore label per gli sore (mantiene il match alle posizioni)
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        # ritorna gli indici per sortare y_score in modo decrescente (dato da [::-1]
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]

        # sorto effettivamente il vettore di probabilities (e di label True False)
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]

        # y_score typically has many tied values. Here we extract the indices associated with the distinct values.
        # We also concatenate a value for the end of the curve.

        # con np.diff() facciamo emergere i valori di probabilità molto diversi dagli altri (generalmente simili)
        # con np.where() selezioniamo gli indici solo nei valori diversi a True (tutti i diversi da 0)
        distinct_value_indices = np.where(np.diff(y_score))[0]
        # np.r_ attacca alla fine del vettore distinct_value_indices il valore di y_true.size - 1
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        # per ogni step conta quanti true positivi ci sono. Sarà nella forma: 0,0,0,0,1,2,3,4
        # a questo accede mediante gli indici dei valori diversi tra loro
        tps = self.stable_cumsum(y_true)[threshold_idxs]
        # add one because of zero-based indexing
        fps = 1 + threshold_idxs - tps

        thresholds = y_score[threshold_idxs]

        recall = tps / tps[-1]

        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)  # [last_ind::-1]
        recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

        cutoff = np.argmin(np.abs(recall - recall_level))

        return fps[cutoff] / (np.sum(np.logical_not(y_true)))  # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

    def stable_cumsum(self, arr, rtol=1e-05, atol=1e-08):
        """Use high precision for cumsum and check that final value matches sum
        Parameters
        ----------
        arr : array-like
            To be cumulatively summed as flat
        rtol : float
            Relative tolerance, see ``np.allclose``
        atol : float
            Absolute tolerance, see ``np.allclose``
        """
        out = np.cumsum(arr, dtype=np.float64)
        expected = np.sum(arr, dtype=np.float64)
        if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
            raise RuntimeError('cumsum was found to be unstable: '
                               'its last element does not correspond to sum')
        return out

    def to_str(self, results):
        string = "\n"
        for k, v in results.items():
            if k != "Class IoU" and k != "Class Acc" and k != "Confusion Matrix":
                string += "%s: %f\n" % (k, v)

        string += 'Class IoU:\n'
        for k, v in results['Class IoU'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        string += 'Class Acc:\n'
        for k, v in results['Class Acc'].items():
            string += "\tclass %d: %s\n" % (k, str(v))

        return string

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def get_results(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
            - auroc
            - aupr
            - fpr95
        """
        auroc = 0
        aupr = 0
        fpr = 0

        EPS = 1e-6
        hist = self.confusion_matrix

        gt_sum = hist.sum(axis=1)
        mask = (gt_sum != 0)
        diag = np.diag(hist)

        acc = diag.sum() / hist.sum()
        acc_cls_c = diag / (gt_sum + EPS)
        acc_cls = np.mean(acc_cls_c[mask])
        iu = diag / (gt_sum + hist.sum(axis=0) - diag + EPS)
        mean_iu = np.mean(iu[mask])
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(range(self.n_classes), [iu[i] if m else "X" for i, m in enumerate(mask)]))
        cls_acc = dict(zip(range(self.n_classes), [acc_cls_c[i] if m else "X" for i, m in enumerate(mask)]))

        if self.total_test_images:
            auroc = self.auroc / self.total_test_images * 100.
            aupr = self.aupr / self.total_test_images * 100.
            fpr = self.fpr / self.total_test_images * 100.

        return {
            "Total samples": self.total_samples,
            "Overall Acc": acc,
            "Mean Acc": acc_cls,
            "FreqW Acc": fwavacc,
            "Mean IoU": mean_iu,
            "Class IoU": cls_iu,
            "Class Acc": cls_acc,
            "Confusion Matrix": self.confusion_matrix_to_fig(),
            "AUROC": auroc,
            "AUPR": aupr,
            "FPR95": fpr,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
        self.total_samples = 0
        self.auroc = 0
        self.aupr = 0
        self.fpr = 0
        self.total_test_images = 0

    def synch(self, device):
        # collect from multi-processes
        confusion_matrix = torch.tensor(self.confusion_matrix).to(device)
        samples = torch.tensor(self.total_samples).to(device)
        auroc = torch.tensor(self.auroc).to(device)
        aupr = torch.tensor(self.aupr).to(device)
        fpr = torch.tensor(self.fpr).to(device)
        test_images = torch.tensor(self.total_test_images).to(device)

        torch.distributed.reduce(confusion_matrix, dst=0)
        torch.distributed.reduce(samples, dst=0)
        torch.distributed.reduce(auroc, dst=0)
        torch.distributed.reduce(aupr, dst=0)
        torch.distributed.reduce(fpr, dst=0)
        torch.distributed.reduce(test_images, dst=0)

        torch.distributed.barrier()

        if torch.distributed.get_rank() == 0:
            self.confusion_matrix = confusion_matrix.cpu().numpy()
            self.total_samples = samples.cpu().numpy()
            self.auroc = auroc.cpu().numpy()
            self.aupr = aupr.cpu().numpy()
            self.fpr = fpr.cpu().numpy()
            self.total_test_images = test_images.cpu().numpy()

    def confusion_matrix_to_fig(self):
        cm = self.confusion_matrix.astype('float') / (self.confusion_matrix.sum(axis=1) + 0.000001)[:, np.newaxis]
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)

        ax.set(title=f'Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')

        fig.tight_layout()
        return fig

class AverageMeter(object):
    """Computes average values"""

    def __init__(self):
        self.book = dict()

    def reset_all(self):
        self.book.clear()

    def reset(self, id):
        item = self.book.get(id, None)
        if item is not None:
            item[0] = 0
            item[1] = 0

    def update(self, id, val):
        record = self.book.get(id, None)
        if record is None:
            self.book[id] = [val, 1]
        else:
            record[0] += val
            record[1] += 1

    def get_results(self, id):
        record = self.book.get(id, None)
        assert record is not None
        return record[0] / record[1]
