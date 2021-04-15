import os
import numpy as np
from math import log10
from collections import OrderedDict
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn import metrics
import matplotlib.pyplot as plt

from scipy.special import entr
import pdb
import cv2
import pandas as pd

from networks import get_generator
from networks.networks_classify import gaussian_weights_init
from models.utils import AverageMeter, get_scheduler, psnr, mse, get_nonlinearity
from skimage.metrics import structural_similarity as compare_ssim

class_names = ['Atelectasis', 'Cardiomegaly', 'Effusion', 'Infiltration', 'Mass',
              'Nodule', 'Pneumonia', 'Pneumothorax', 'Consolidation', 'Edema',
              'Emphysema', 'Fibrosis', 'Pleural_Thickening', 'Hernia']


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


class CNNModel(nn.Module):
    def __init__(self, opts):
        super(CNNModel, self).__init__()

        self.results = {}
        self.loss_names = []
        self.networks = []
        self.optimizers = []

        # set default loss flags
        loss_flags = ("w_img_BCE")
        for flag in loss_flags:
            if not hasattr(opts, flag): setattr(opts, flag, 0)

        self.is_train = True if hasattr(opts, 'lr') else False

        self.net_G = get_generator(opts.net_G, opts)
        self.networks.append(self.net_G)

        if self.is_train:
            self.loss_names += ['loss_G_BCE']
            self.optimizer_G = torch.optim.Adam(self.net_G.parameters(),
                                                lr=opts.lr,
                                                betas=(opts.beta1, opts.beta2),
                                                weight_decay=opts.weight_decay)
            self.optimizers.append(self.optimizer_G)

        self.criterion = nn.BCEWithLogitsLoss()

        self.opts = opts

    def setgpu(self, gpu_ids):
        self.device = torch.device('cuda:{}'.format(gpu_ids[0]))

    def initialize(self):
        [net.apply(gaussian_weights_init) for net in self.networks]

    def set_scheduler(self, opts, epoch=-1):
        self.schedulers = [get_scheduler(optimizer, opts, last_epoch=epoch) for optimizer in self.optimizers]

    def set_input(self, data):
        self.img = data['img'].to(self.device).float()
        self.labels_gt = data['label'].to(self.device).float()

    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, name))
        return errors_ret

    def set_epoch(self, epoch):
        self.curr_epoch = epoch

    def forward(self):
        inp = self.img
        inp.requires_grad_(True)
        self.labels_pred, self.heatmaps = self.net_G(inp)

    def update_G(self):
        loss_G_BCE = 0
        self.optimizer_G.zero_grad()
        loss_G_BCE = self.criterion(self.labels_pred, self.labels_gt)
        self.loss_G_BCE = loss_G_BCE.item()

        total_loss = loss_G_BCE
        total_loss.backward()
        self.optimizer_G.step()

    def optimize(self):
        self.loss_G_BCE = 0
        self.forward()
        self.update_G()

    @property
    def loss_summary(self):
        message = ''
        message += 'G_BCE: {:.4e} '.format(self.loss_G_BCE)

        return message

    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = {:7f}'.format(lr))

    def save(self, filename, epoch, total_iter):

        state = {}
        state['net_G'] = self.net_G.module.state_dict()
        state['opt_G'] = self.optimizer_G.state_dict()

        state['epoch'] = epoch
        state['total_iter'] = total_iter

        torch.save(state, filename)
        print('Saved {}'.format(filename))

    def resume(self, checkpoint_file, train=True):
        checkpoint = torch.load(checkpoint_file, map_location='cuda:0')

        self.net_G.module.load_state_dict(checkpoint['net_G'])
        if train:
            self.optimizer_G.load_state_dict(checkpoint['opt_G'])

        print('Loaded {}'.format(checkpoint_file))

        return checkpoint['epoch'], checkpoint['total_iter']

    def evaluate(self, loader):
        val_bar = tqdm(loader)

        labels_pred_all = []
        labels_gt_all = []
        heatmap_images = []

        for data in val_bar:
            self.set_input(data)
            self.forward()

            labels_pred_all.append(self.labels_pred[0])
            labels_gt_all.append(self.labels_gt[0])

            heatmap_images.append(self.heatmaps[0].cpu())

        labels_pred_all = torch.stack(labels_pred_all).squeeze().cpu().numpy()
        labels_gt_all = torch.stack(labels_gt_all).squeeze().cpu().numpy()
        all_Ap, mAp = metric_mAp(labels_pred_all, labels_gt_all)
        all_AUC, mAUC = metric_mAUC(labels_pred_all, labels_gt_all)

        self.all_Ap = all_Ap
        self.mAp = mAp
        self.all_AUC = all_AUC
        self.mAUC = mAUC

        n_class = labels_gt_all.shape[1]
        message = ''
        for i in range(n_class):
            message += 'Ap_{}: {:4f} '.format(class_names[i], all_Ap[i])
        message += 'mAp: {:4f} '.format(mAp)
        print(message)

        for i in range(n_class):
            message += 'AUC_{}: {:4f} '.format(class_names[i], all_AUC[i])
        message += 'mAUC: {:4f} '.format(mAUC)
        print(message)

        cmap = get_cmap(n_class)
        plt.title('Receiver Operating Characteristic')

        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_class):
            fpr[i], tpr[i], _ = metrics.roc_curve(labels_gt_all[:, i], labels_pred_all[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

            plt.plot(fpr[i], tpr[i], color=cmap(i), label="AUC {:s} = {:0.2f}".format(class_names[i], roc_auc[i]))

        plt.legend(loc='lower right')
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig("roc.png")

        self.results['labels_pred_all'] = labels_pred_all
        self.results['labels_gt_all'] = labels_gt_all
        self.results['heatmaps'] = torch.stack(heatmap_images).squeeze().numpy()

    def generate_heatmaps(self):
        file_list = []
        with open("../Data/ChestXray14/labels/test_list_boxes.txt", "r") as in_file:
            lines = in_file.readlines()
            for line in lines:
                name = line.split(" ")[0]
                file_list.append(name)
        bbox_data = pd.read_csv("../Data/ChestXray14/BBox_List_2017.csv")
        bbox_data = bbox_data.loc[:, :'h]']

        bbox_data.rename(columns={'Bbox [x': 'x',
                                  'h]': 'h',
                                  'Image Index': 'images',
                                  'Finding Label': 'labels'},
                         inplace=True)

        bbox_data.labels.replace(to_replace='Infiltrate',
                                 value='Infiltration',
                                 inplace=True)
        for image_id in range(self.results["heatmaps"].shape[0]):
            gt_boxes = bbox_data.loc[bbox_data["images"] == file_list[image_id]]
            gt_labels = gt_boxes[['labels']]
            gt_boxes = gt_boxes[['x', 'y', 'w', 'h']] / 4
            class_inds = []
            for gt_label in gt_labels.values:
                class_inds.append(class_names.index(gt_label))

            labels = torch.sigmoid(torch.from_numpy(self.results['labels_gt_all'][image_id]))
            labels = labels.detach().cpu().numpy()

            labels = np.where(labels > 0.5)[0]
            labels = sorted(labels)
            class_inds = sorted(class_inds)

            for i, cls_index in enumerate(class_inds):
                if cls_index in labels:
                    heatmap = self.results["heatmaps"][image_id, cls_index, :, :]
                    img_min = np.min(heatmap)
                    img_max = np.max(heatmap)
                    heatmap = (heatmap - img_min) / (img_max - img_min)

                    heatmap = cv2.resize(heatmap, (256, 256))
                    heatmap_to_draw = heatmap.copy()
                    # Binarize the activations
                    _, heatmap = cv2.threshold(heatmap, 0.7, 1, type=cv2.THRESH_BINARY)
                    heatmap = np.uint8(255 * heatmap)

                    img = cv2.imread("../Data/ChestXray14/images/" + file_list[image_id])
                    img = cv2.resize(img, (256, 256))

                    contour, _ = cv2.findContours(heatmap, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                    rect = cv2.boundingRect(contour[0])
                    heatmap_to_draw = np.uint8(255 * heatmap_to_draw)
                    heatmap_to_draw = cv2.applyColorMap(heatmap_to_draw, cv2.COLORMAP_JET)
                    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_to_draw, 0.4, 0)
                    box = gt_boxes.values[i]
                    superimposed_img = cv2.rectangle(superimposed_img, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])), color=(0, 255, 0),
                                                         thickness=2)
                    superimposed_img = cv2.rectangle(superimposed_img, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color=(0, 0, 255),
                                 thickness=2)

                    cv2.imwrite("outputs/test_ChestXray14_densenetADA/heatmaps/"+"target_" + str(class_names[cls_index])  + "_" + file_list[image_id], superimposed_img)
                    print("File ", "outputs/test_ChestXray14_densenetADA/heatmaps/"+"target_" + str(class_names[cls_index])  + "_" + file_list[image_id], "is written")


def metric_mAp(output, target):
    """ Calculation of mAp """
    output_np = output
    target_np = target

    num_class = target.shape[1]
    all_ap = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        TP = np.sum(gt_cls * pred_cls)
        FP = np.sum(gt_cls * (1 - pred_cls))

        if TP == 0 and FP == 0:
            continue
        else:
            pred_cls = pred_cls - 1e-5 * gt_cls
            ap = average_precision_score(gt_cls, pred_cls, average=None)

        all_ap.append(ap)

    mAP = np.mean(all_ap)
    return all_ap, mAP


def metric_mAUC(output, target):
    """ Calculation of ROC AUC """
    output_np = output
    target_np = target

    num_class = target.shape[1]
    all_roc_auc = []
    for cid in range(num_class):
        gt_cls = target_np[:, cid].astype('float32')
        pred_cls = output_np[:, cid].astype('float32')

        TP = np.sum(gt_cls * pred_cls)
        FP = np.sum(gt_cls * (1 - pred_cls))

        if TP == 0 and FP == 0:
            continue
        else:
            pred_cls = pred_cls - 1e-5 * gt_cls
            roc_auc = roc_auc_score(gt_cls, pred_cls, average='weighted')

        all_roc_auc.append(roc_auc)

    mROC_AUC = np.mean(all_roc_auc)
    return all_roc_auc, mROC_AUC