
import argparse
import torch
from torchvision import transforms
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger

def dataset_loader(dataset_name,size,n_classes,batch_size):

    train_images = f'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/{dataset_name}/data/images/train'
    test_images = f'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/{dataset_name}/data/images/test'
    train_labled = f'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/{dataset_name}/data/labeled/train'
    test_labeled = f'E:/Data/Dataset/Processed_image/lungs-segmentation-dataset/{dataset_name}/data/labeled/test'

    ### Loader
    compose = transforms.Compose([
        Rescale((size,size)),
        ToTensor()
            ])

    train_datasets = SegmentationDataset(train_images, train_labled, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8)

    test_datasets = SegmentationDataset(test_images,test_labeled, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True,num_workers=8)
    
    return train_loader,test_loader

def training(checkpoints_path,args):    
    model_name = args.model_name
    device = args.device
    batch_size = args.batch_size
    n_classes = args.num_classes
    num_epochs = args.epochs
    size = args.size
    pretrained = args.pretrained
    fixed_feature = args.fixed_feature


    train_loader,test_loader = dataset_loader(args.dataset,size,n_classes,batch_size)
    
    logger = Logger(model_name=model_name, data_name=args.dataset)
    
    model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                pretrained=pretrained,
                                                fixed_feature=fixed_feature)





    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())


    if checkpoints_path:

        checkpoint = torch.load(checkpoints_path)
        best_metrics = checkpoint['best_metric']
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])


    else :
        start_epoch = 0
        best_metrics = 0



    trainer = Trainer(start_epoch,best_metrics,args.dataset,model, optimizer, logger, num_epochs, train_loader, test_loader)
    trainer.train()



def main():
    # All the necessary argument
    parser = argparse.ArgumentParser(description="Fully Convolutional Netwrok With Backbone")
    
    parser.add_argument('--dataset', type=str, default="china_xrays_dataset",
                    help='dataset path. For this create the json file')
    
    
    parser.add_argument('--resume', type=str, default=None,
                    help='put the path to resuming file if needed')
    
    parser.add_argument('--model_name', type=str, default="fcn16_resnet101",
                    help='model name available')
    
    parser.add_argument('--epochs',type=int, default=500,
                        help='Number of iteration while training')
    
    parser.add_argument('--batch_size',type=int, default=4,
                        help='input batch size for training (default = 1)')
    
    parser.add_argument('--num_worker',type=int, default=2,
                        help='the number of worker threads to use for loading data in parallel (default = 1)')
    
    parser.add_argument('--num_classes',type=int, default=2,
                        help='how many organ to segment include background')  
    
    parser.add_argument('--size', type=int, default=256,
                    help='dataset resize')

    parser.add_argument('--device', type=str, default="cuda",
                    help='dataset resize')

    parser.add_argument('--pretrained', type=bool, default=True,
                    help='backbone pretrained weight activate')
    
    parser.add_argument('--fixed_feature', type=bool, default=False,
                    help='fixed_feature activate')


    args = parser.parse_args()


    training(args.resume,args)

if __name__ == "__main__":
    main()