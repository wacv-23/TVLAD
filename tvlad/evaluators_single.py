from __future__ import print_function, absolute_import
import torch
import torch.nn.functional as F
import time
from collections import OrderedDict
import numpy as np

from .utils.meters import AverageMeter
from .utils.rerank import re_ranking
from .utils import to_torch


def extract_cnn_feature(model, inputs, vlad=True, gpu=None):
    model.eval()
    if gpu == None:
        inputs = to_torch(inputs)
    else:
        inputs = to_torch(inputs).cuda(gpu)
    outputs = model(inputs)
    if (isinstance(outputs, list) or isinstance(outputs, tuple)):
        x_pool, x_vlad = outputs
        if vlad:
            outputs = F.normalize(x_vlad, p=2, dim=-1)
        else:
            outputs = F.normalize(x_pool, p=2, dim=-1)
    else:
        outputs = F.normalize(outputs, p=2, dim=-1)
    return outputs


def extract_features(model, data_loader, dataset, print_freq=10,
                vlad=True, pca=None, gpu=None):
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    features = []

    if (pca is not None):
        pca.load(gpu=gpu)

    end = time.time()
    with torch.no_grad():
        for i, (imgs, fnames, _, _, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            outputs = extract_cnn_feature(model, imgs, vlad, gpu=gpu)

            if (pca is not None):
                outputs = pca.infer(outputs)
            outputs = outputs.data.cpu()
            features.append(outputs)

            batch_time.update(time.time() - end)
            end = time.time()

            if ((i + 1) % print_freq == 0):
                print('Extract Features: [{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      .format(i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg))

    if (pca is not None):
        del pca

    bc_features = torch.cat(features).cuda(gpu)
    features_dict = OrderedDict()
    for fname, output in zip(dataset, bc_features.cpu()):
        features_dict[fname[0]] = output
    del bc_features, features
    return features_dict


def pairwise_distance(features, query=None, gallery=None, metric=None):
    if query is None and gallery is None:
        n = len(features)
        x = torch.cat(list(features.values()))
        x = x.view(n, -1)
        if metric is not None:
            x = metric.transform(x)
        dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True) * 2
        dist_m = dist_m.expand(n, n) - 2 * torch.mm(x, x.t())
        return dist_m, None, None

    print ("===> Start calculating pairwise distances")
    x = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in query], 0)
    y = torch.cat([features[f].unsqueeze(0) for f, _, _, _ in gallery], 0)

    m, n = x.size(0), y.size(0)
    x = x.view(m, -1)
    y = y.view(n, -1)
    if metric is not None:
        x = metric.transform(x)
        y = metric.transform(y)
    dist_m = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(m, n) + \
           torch.pow(y, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_m.addmm_(1, -2, x, y.t())
    return dist_m, x.numpy(), y.numpy()

def spatial_nms(pred, db_ids, topN):
    assert(len(pred)==len(db_ids))
    pred_select = pred[:topN]
    pred_pids = [db_ids[i] for i in pred_select]
    # find unique
    seen = set()
    seen_add = seen.add
    pred_pids_unique = [i for i, x in enumerate(pred_pids) if not (x in seen or seen_add(x))]
    return [pred_select[i] for i in pred_pids_unique]


def evaluate_all(distmat, gt, gallery, recall_topk=[1, 5, 10, 15, 20, 25], nms=False):
    sort_idx = np.argsort(distmat, axis=1)
    del distmat
    db_ids = [db[1] for db in gallery]

    print("===> Start calculating recalls")
    correct_at_n = np.zeros(len(recall_topk))

    for qIx, pred in enumerate(sort_idx):
        if (nms):
            pred = spatial_nms(pred.tolist(), db_ids, max(recall_topk)*12)
  
        for i, n in enumerate(recall_topk):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    recalls = correct_at_n / len(gt)
    del sort_idx

    print('Recall Scores:')
    for i, k in enumerate(recall_topk):
        print('  top-{:<4}{:12.1%}'.format(k, recalls[i]))
    return recalls


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def evaluate(self, query_loader, dataset, query, gallery, ground_truth, gallery_loader=None, \
                    vlad=True, pca=None, rerank=False, gpu=None, \
                    nms=False, rr_topk=25, lambda_value=0):
        if (gallery_loader is not None):
            features = extract_features(self.model, query_loader, query,
                                        vlad=vlad, pca=pca, gpu=gpu)
            features_db = extract_features(self.model, gallery_loader, gallery,
                                        vlad=vlad, pca=pca, gpu=gpu)
            features.update(features_db)
        else:
            features = extract_features(self.model, query_loader, dataset,
                            vlad=vlad, pca=pca, gpu=gpu)

        distmat, _, _ = pairwise_distance(features, query, gallery)
        recalls = evaluate_all(distmat, ground_truth, gallery, nms=nms)
        if (not rerank):
            return recalls

        print('Applying re-ranking ...')
        distmat_gg, _, _ = pairwise_distance(features, gallery, gallery)
        distmat_qq, _, _ = pairwise_distance(features, query, query)
        distmat = re_ranking(distmat.numpy(), distmat_qq.numpy(), distmat_gg.numpy(),
                            k1=rr_topk, k2=1, lambda_value=lambda_value)

        return evaluate_all(distmat, ground_truth, gallery, nms=nms)
