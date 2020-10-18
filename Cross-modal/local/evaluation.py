# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Evaluation"""

from __future__ import print_function
import os

import sys
from data import get_test_loader
import time
import numpy as np
from vocab import Vocabulary, deserialize_vocab  # NOQA
import torch
from model import SCAN, joint_embedding_loss,direct_attention_score
from collections import OrderedDict
import time
from torch.autograd import Variable
import scipy.io as sio

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)


class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)


def encode_data(model, data_loader, log_step=10, logging=print):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None
    image_entity_idxs = None
    caps_entity_idxs = None

    max_n_word = 0
    for i, (images, captions, lengths, image_entity_idx, caps_entity_idx, ids) in enumerate(data_loader):
        #print(lengths)
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, image_entity_idx, caps_entity_idx, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger
                
        # compute the embeddings
        img_emb, cap_emb, cap_len = model.forward_emb(images, captions, lengths, volatile=True)

        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
            image_entity_idxs = np.zeros((len(data_loader.dataset), 36))
            caps_entity_idxs = np.zeros((len(data_loader.dataset), 50))
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        image_entity_idxs[ids] = image_entity_idx.data.cpu().numpy().copy()
        caps_entity_idxs[ids] = caps_entity_idx.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = cap_len[j]

        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, image_entity_idx, caps_entity_idx, cap_len)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del images, captions
    return img_embs, cap_embs, cap_lens, image_entity_idxs, caps_entity_idxs


def evalrank(model_path, data_path=None, split='dev', fold5=False):
    """
    Evaluate a trained model on either dev or test. If `fold5=True`, 5 fold
    cross-validation is done (only for MSCOCO). Otherwise, the full data is
    used for evaluation.
    """
    # load model and options
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']
    print(opt)
    if data_path is not None:
        opt.data_path = data_path
    # load vocabulary used by the model
    vocab = deserialize_vocab(os.path.join(opt.vocab_path, '%s_vocab.json' % opt.data_name))
    opt.vocab_size = len(vocab)

    # construct model
    model = SCAN(opt)

    # load model state
    model.load_state_dict(checkpoint['model'])

    print('Loading dataset')
    data_loader = get_test_loader(split, opt.data_name, vocab,
                                  opt.batch_size, opt.workers, opt)

    print('Computing results...')
    img_embs, cap_embs, cap_lens, image_entity_idxs, caps_entity_idxs = encode_data(model, data_loader)
    print('Images: %d, Captions: %d' %
          (img_embs.shape[0] , cap_embs.shape[0]))
          
    img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
    print('Images: ', img_embs.shape)
    print('Captions: ', cap_embs.shape)
    
    start = time.time()
    sims = compute_sims(img_embs, cap_embs, cap_lens, image_entity_idxs, caps_entity_idxs, opt, shard_size=128)
    print(sims[:20,:4])
    end = time.time()
    print("calculate similarity time:", end-start)
    
    print('Saving results...')
    sio.savemat('sim-large.mat',{'similarity':sims})
    print('Saving success...')
    
    r, rt = i2t(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ri, rti = t2i(img_embs, cap_embs, cap_lens, sims, return_ranks=True)
    ar = (r[0] + r[1] + r[2]) / 3
    ari = (ri[0] + ri[1] + ri[2]) / 3
    rsum = r[0] + r[1] + r[2] + ri[0] + ri[1] + ri[2]
    print("rsum: %.1f" % rsum)
    print("Average i2t Recall: %.1f" % ar)
    print("Image to text: %.1f %.1f %.1f %.1f %.1f" % r)
    print("Average t2i Recall: %.1f" % ari)
    print("Text to image: %.1f %.1f %.1f %.1f %.1f" % ri)
    
    torch.save({'rt': rt, 'rti': rti}, 'ranks.pth.tar')


def softmax(X, axis):
    """
    Compute the softmax of each element along an axis of X.
    """
    y = np.atleast_2d(X)
    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    # exponentiate y
    y = np.exp(y)
    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)
    # finally: divide elementwise
    p = y / ax_sum
    return p

def compute_sims(images, captions, caplens, image_entity_idxs, caps_entity_idxs, opt, shard_size=128):
    """
    Computer pairwise t2i distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    d = np.zeros((len(captions),len(images)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), requires_grad=False).cuda()
            image_entity_idx = Variable(torch.from_numpy(image_entity_idxs[im_start:im_end]), requires_grad=False).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), requires_grad=False).cuda()
            caps_entity_idx = Variable(torch.from_numpy(caps_entity_idxs[cap_start:cap_end]), requires_grad=False).cuda()
            l = caplens[cap_start:cap_end]
            latent_scores = joint_embedding_loss(im, s, l, opt).cuda()
            direct_scores = direct_attention_score(im, s, image_entity_idx, caps_entity_idx, l, opt).cuda()
            sim = latent_scores + direct_scores
            d[cap_start:cap_end, im_start:im_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    
    # --> (5N(caption), N(image))
    sims = sims.T
    
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)


    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


