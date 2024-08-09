import os
import torch
import random
import numpy as np
import torch
import logging
import os
from sklearn import metrics
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import torch.nn as nn
import random


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def loss_mse(pred_recon, imgs, mask):
    # loss = torch.pow((pred_recon-imgs)*(1-mask), 2).sum()/((1-mask).sum())
    # loss = torch.pow((pred_recon-imgs), 2).mean()
    loss = torch.abs(pred_recon - imgs).mean()
    return loss


def acc_tpr_tnr(y_pre, y):
    true = y_pre == y
    acc = true.sum() / y_pre.shape[0]
    tpr = true[y == 1].sum() / (y == 1).sum()
    tnr = true[y == 0].sum() / (y == 0).sum()
    return acc.item() * 100, tpr.item() * 100, tnr.item() * 100


def truepositive_filter(pred, gt, filter_ratio):
    # pred [N, 2], softmax
    pred_1 = pred[:, 1]
    positives = pred_1[gt == 1]
    if len(positives) > 0:
        threshold = positives.min()
    else:
        # return [(torch.Tensor([0])/0)[0]]*len(filter_ratio)
        return [0] * len(filter_ratio)

    positives_sort = np.sort(positives.numpy())[::-1]
    out = []
    thresholds = []
    for ratio in filter_ratio:
        threshold = positives_sort[round(ratio * positives_sort.shape[0]) - 1]
        tnr = (pred_1[gt == 0] < threshold).sum() / (gt == 0).sum()
        out.append(tnr.item() * 100)
        thresholds.append(float(threshold))
    return out, thresholds


def truepositive_filter_test(pred, gt, filter_ratio):
    # pred [N, 2], softmax
    pred_1 = pred[:, 1]
    positives = pred_1[gt == 1]
    if len(positives) > 0:
        threshold = positives.min()
    else:
        # return [(torch.Tensor([0])/0)[0]]*len(filter_ratio)
        return [0] * len(filter_ratio)

    positives_sort = np.sort(positives.numpy())[::-1]
    out = []
    thresholds = []
    for ratio in filter_ratio:
        threshold = positives_sort[round(ratio * positives_sort.shape[0]) - 1]
        tnr = (pred_1[gt == 0] < threshold).sum() / (gt == 0).sum()
        out.append(tnr.item() * 100)
        thresholds.append(float(threshold))
    return out, thresholds


def auc_score(pred, gt):
    pred_1 = pred[:, 1].numpy()
    fpr, tpr, thresholds = metrics.roc_curve(gt.numpy(), pred_1)
    auc = metrics.auc(fpr, tpr)
    return auc * 100


def construct_log(log_pth, only_file):
    # logger = logging.getLogger(__name__)
    # logger.setLevel(level = logging.INFO)
    logging.root.setLevel(logging.INFO)

    handler = logging.FileHandler(log_pth)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logging.root.addHandler(handler)

    if not only_file:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logging.root.addHandler(console_handler)

        # return logger


def print_args(args, logger):
    logger.info('-------- args -----------')
    for k, v in vars(args).items():
        logger.info('%s: ' % k + str(v))
    logger.info('-------------------------')


def _cumulative_sum_threshold(values, percentile):
    # given values should be non-negative
    assert percentile >= 0 and percentile <= 100, (
        "Percentile for thresholding must be " "between 0 and 100 inclusive."
    )
    sorted_vals = np.sort(values.flatten())
    cum_sums = np.cumsum(sorted_vals)
    threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
    return sorted_vals[threshold_id]


# def gradient_visualization(model, imgs, gts, mean, std, _logger):
#     mean = mean.view(1, 3, 1, 1)
#     std = std.view(1, 3, 1, 1)
#     imgs_trans = imgs * std + mean
#
#     # https://captum.ai/tutorials/Resnet_TorchVision_Interpret
#     from captum.attr import IntegratedGradients, NoiseTunnel
#     from captum.attr import visualization as viz
#     from matplotlib.colors import LinearSegmentedColormap
#
#     default_cmap = LinearSegmentedColormap.from_list('custom blue',
#                                                      [(0, '#ffffff'),
#                                                       (0.25, '#000000'),
#                                                       (1, '#000000')], N=256)
#
#     output = model(imgs)
#     output = F.softmax(output, dim=1)
#     prediction_score, pred_label_idx = torch.topk(output, 1)
#
#     save_dir = 'captum_visualization'
#     for i in range(imgs.shape[0]):
#         integrated_gradients = IntegratedGradients(model)
#         noise_tunnel = NoiseTunnel(integrated_gradients)
#         attributions_ig_nt = noise_tunnel.attribute(imgs[i:i + 1], nt_samples=6, nt_type='smoothgrad_sq',
#                                                     target=pred_label_idx[i:i + 1])
#
#         attri_img = np.transpose(attributions_ig_nt.squeeze().cpu().detach().numpy(), (1, 2, 0))
#         orig_img = np.transpose(imgs_trans[i].cpu().detach().numpy(), (1, 2, 0))
#         _ = viz.visualize_image_attr_multiple(attri_img,
#                                               orig_img,
#                                               ["original_image", "heat_map"],
#                                               ["all", "positive"],
#                                               cmap=default_cmap,
#                                               show_colorbar=True)
#
#         img_name = '%d_pred%d_gt%d' % (i, pred_label_idx[i].item(), gts[i].item())
#         plt.savefig(os.path.join(save_dir, img_name + '.pdf'))
#
#         attri_img_b = attri_img
#         attri_img_b = attri_img_b.sum(2)
#         attri_img_b[attri_img_b < 0] = 0
#         threshold = _cumulative_sum_threshold(attri_img_b, 100 - 2)
#         attri_img_b = np.clip(attri_img_b / threshold, 0, 1)
#
#         attri_img_heat = cv2.applyColorMap(np.uint8(255 * attri_img_b),
#                                            cv2.COLORMAP_JET)  # https://huaweicloud.csdn.net/63808affdacf622b8df8a235.html
#         matrix_k = attri_img_b[:, :, None] * 0.9
#
#         orig_img_cv2 = np.uint8(orig_img * 255)
#         orig_img_cv2 = cv2.cvtColor(orig_img_cv2, cv2.COLOR_BGR2RGB)
#
#         attri_blend = matrix_k * attri_img_heat + (1 - matrix_k) * orig_img_cv2
#         cv2.imwrite(os.path.join(save_dir, img_name + '.png'), attri_blend)
#
#         _logger.info('gradient_visualization: %d/%d: %s' % (i, imgs.shape[0], img_name))


def model_in_convert(model, model_in_chans, ori_in_chans, pretrained_cfg):
    input_conv_name = pretrained_cfg.first_conv
    module = getattr(model, input_conv_name)
    assert not getattr(module, 'bias')
    assert type(module) == nn.Conv2d

    if model_in_chans != 3:
        assert model_in_chans > 3 and model_in_chans % 3 == 0
        if ori_in_chans != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = model_in_chans // 3
            conv_weight = getattr(module, 'weight')
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)

            new_conv = nn.Conv2d(model_in_chans, module.out_channels, stride=module.stride,
                                 kernel_size=module.kernel_size,
                                 padding=module.padding, bias=False)
            new_conv.weight.data = conv_weight
            setattr(model, input_conv_name, new_conv)
cudnn_deterministic = True


def seed_everything(seed=0):
    """Fix all random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = cudnn_deterministic


def print_summary(acc_taw, acc_tag, forg_taw, forg_tag):
    """Print summary of results"""
    for name, metric in zip(['TAw Acc', 'TAg Acc', 'TAw Forg', 'TAg Forg'], [acc_taw, acc_tag, forg_taw, forg_tag]):
        print('*' * 108)
        print(name)
        for i in range(metric.shape[0]):
            print('\t', end='')
            for j in range(metric.shape[1]):
                print('{:5.1f}% '.format(100 * metric[i, j]), end='')
            if np.trace(metric) == 0.0:
                if i > 0:
                    print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i].mean()), end='')
            else:
                print('\tAvg.:{:5.1f}% '.format(100 * metric[i, :i + 1].mean()), end='')
            print()
    print('*' * 108)
