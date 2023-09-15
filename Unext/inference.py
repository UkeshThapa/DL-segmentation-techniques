import argparse
import os
from glob import glob
from secrets import choice

import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import archs
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter
from albumentations import RandomRotate90,Resize
import time
from archs import UNext


# def parse_args():
#     parser = argparse.ArgumentParser()

#     parser.add_argument('--name', default='china_xrays_dataset_UNext_woDS',
#                         help='model name',choices=['china_xrays_dataset_UNext_woDS'])

#     args = parser.parse_args()

#     return args


def main():
    # args = parse_args()

    model = ['darwinlungs_small']
    test_datasets = ['nih_xrays_dataset','china_xrays_dataset','japan_xrays_dataset','montgomery_xrays_dataset','darwinlungs','u5_dataset']
    


    for names in model:
        print(f"\nLoading --->{names} model")

        with open('models/%s_UNext_model/config.yml' % names, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        print('-'*20)
        for key in config.keys():
            print('%s: %s' % (key, str(config[key])))
        print('-'*20)

        cudnn.benchmark = True

        print("=> creating model %s" % config['arch'])
        model = archs.__dict__[config['arch']](config['num_classes'],
                                            config['input_channels'],
                                            config['deep_supervision'])

        model = model.cuda()


        model.load_state_dict(torch.load('models/%s/model.pth' %
                                        config['name']))
        print(model.state_dict())
        model.eval()

        for datasets in  test_datasets:

            print(f"\nTesting with ----> {datasets}")
            # Change by UK
            img_ids = []
            with open(os.path.join('E:\Data\Dataset\Processed_image\lungs-segmentation-dataset',datasets,'split_dataset','test.txt'), "r") as f:
                lines = f.read().splitlines()

            for ii, line in enumerate(lines):
                img_ids.append(line)



            test_transform = Compose([
                Resize(config['input_h'], config['input_w']),
                transforms.Normalize(),
            ])

            test_dataset = Dataset(
                img_ids= img_ids,
                img_dir=os.path.join('E:\Data\Dataset\Processed_image\lungs-segmentation-dataset',datasets, 'images'),
                mask_dir=os.path.join('E:\Data\Dataset\Processed_image\lungs-segmentation-dataset', datasets, 'masks'),
                img_ext=config['img_ext'],
                mask_ext=config['mask_ext'],
                num_classes=config['num_classes'],
                datasets_name = datasets,
                transform=test_transform)

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'],
                drop_last=False)

            iou_avg_meter = AverageMeter()
            dice_avg_meter = AverageMeter()
            gput = AverageMeter()
            cput = AverageMeter()

            # count = 0
            # for c in range(config['num_classes']):
            SAVE_BASE_PATH = f'E:\\Data\\Dataset\\Results\\Segmentation\\segmentation_Unet\\' 
            if os.path.isdir(SAVE_BASE_PATH+names+'_small'+'_model') is False:         
                os.mkdir(SAVE_BASE_PATH+names+'_small'+'_model')

            paths = SAVE_BASE_PATH+names+'_small'+'_model\\'

            if os.path.isdir(paths+datasets) is False:
                os.mkdir(paths+datasets)
            # os.makedirs(os.path.join('E:\Data\Dataset\Results\Segmentation\segmentation_Unet', config['dataset'],datasets), exist_ok=True)
            with torch.no_grad():
                for input, target, meta in tqdm(test_loader, total=len(test_loader)):
                    input = input.cuda()
                    target = target.cuda()
                    model = model.cuda()
                    # compute output
                    output = model(input)


                    iou,dice = iou_score(output, target)
                    iou_avg_meter.update(iou, input.size(0))
                    dice_avg_meter.update(dice, input.size(0))

                    output = torch.sigmoid(output).cpu().numpy()
                    output[output>=0.5]=1
                    output[output<0.5]=0
# UPDATED BY UKESH
                    gt_paths = f"E:\\Data\\Dataset\\Processed_image\\lungs-segmentation-dataset\\{datasets}\\"
 
                    for i in range(len(output)):
                        for c in range(config['num_classes']):
                            img_name,_,_ =  meta['img_id'][i].partition(".")
                            cv2.imwrite(os.path.join(paths, datasets, img_name+".png"),
                                        (output[i, c] * 255),[cv2.IMWRITE_PNG_BILEVEL, 1])
                            if datasets == 'japan_xrays_dataset' or datasets == 'montgomery_xrays_dataset': 
                                img = cv2.imread(f"{gt_paths}\\masks\\{img_name}.png")
                            else:
                                img = cv2.imread(f"{gt_paths}\\masks\\{img_name}_mask.png")
                            h,w,c = img.shape 

                            preds = cv2.imread(f"{paths}\\{datasets}\\{ img_name}.png") 
                            resized = cv2.resize(preds, (w, h), 0, 0, interpolation = cv2.INTER_NEAREST)
                            gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
                            cv2.imwrite(f"{paths}\\{datasets}\\{img_name}.png" ,gray_image,[cv2.IMWRITE_PNG_BILEVEL, 1])

            print('IoU: %.4f' % iou_avg_meter.avg)
            print('Dice: %.4f' % dice_avg_meter.avg)

            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()