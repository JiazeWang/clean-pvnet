from lib.datasets.dataset_catalog import DatasetCatalog
from lib.config import cfg
import pycocotools.coco as coco
import numpy as np
from lib.utils.pvnet import pvnet_config
import matplotlib.pyplot as plt
from lib.utils import img_utils
import matplotlib.patches as patches
from lib.utils.pvnet import pvnet_pose_utils
from PIL import Image

mean = pvnet_config.mean
std = pvnet_config.std


class Visualizer:

    def __init__(self):
        args = DatasetCatalog.get(cfg.test.dataset)
        self.ann_file = args['ann_file']
        self.coco = coco.COCO(self.ann_file)

    def visualize(self, output, batch, name, high_resolution):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        #inpnew = Image.open(high_resolution)
        kpt_2d = output['kpt_2d'][0].detach().cpu().numpy()
        print(kpt_2d)
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        kpt_3d = np.concatenate([anno['fps_3d'], [anno['center_3d']]], axis=0)
        #K = np.array(anno['K'])
        #print(K)
        #print(K.shape)
        K = np.array([[1.2430691e+03, 0, 5.45181403e+02],
                      [0, 1.23942895e+03, 7.0817400e+02],
                      [0, 0, 1]])

        #pose_gt = np.array(anno['pose'])
        pose_pred = pvnet_pose_utils.pnp(kpt_3d, kpt_2d, K)

        corner_3d = np.array(anno['corner_3d'])
        #corner_2d_gt = pvnet_pose_utils.project(corner_3d, K, pose_gt)
        corner_2d_pred = pvnet_pose_utils.project(corner_3d, K, pose_pred)
        #print(corner_2d_pred)
        _, ax = plt.subplots(1)
        plt.axis('off')
        frame = plt.gca()
        frame.axes.get_yaxis().set_visible(False)
        frame.axes.get_xaxis().set_visible(False)
        ax.imshow(inp)
        #ax.add_patch(patches.Polygon(xy=corner_2d_gt[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=1, edgecolor='g'))
        #ax.add_patch(patches.Polygon(xy=corner_2d_gt[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=1, edgecolor='g'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[0, 1, 3, 2, 0, 4, 6, 2]], fill=False, linewidth=2, edgecolor='r'))
        ax.add_patch(patches.Polygon(xy=corner_2d_pred[[5, 4, 6, 7, 5, 1, 3, 7]], fill=False, linewidth=2, edgecolor='r'))
        ax.figure.savefig(name, bbox_inches='tight')
        #plt.close('all')
        #plt.show()

    def visualize_train(self, output, batch):
        inp = img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0)
        mask = batch['mask'][0].detach().cpu().numpy()
        vertex = batch['vertex'][0][0].detach().cpu().numpy()
        img_id = int(batch['img_id'][0])
        anno = self.coco.loadAnns(self.coco.getAnnIds(imgIds=img_id))[0]
        fps_2d = np.array(anno['fps_2d'])
        plt.figure(0)
        plt.subplot(221)
        plt.imshow(inp)
        plt.subplot(222)
        plt.imshow(mask)
        plt.plot(fps_2d[:, 0], fps_2d[:, 1])
        plt.subplot(224)
        plt.imshow(vertex)
        plt.savefig('test.jpg')
        plt.close(0)
