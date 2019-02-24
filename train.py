# Imports here
import argparse
import helper_train as h


def main(data_dir, save_dir, model_arch, lr, hidden_units, epoch, on_gpu):
    """Train a model on data set of 102 flowers
    
    The data set for the training can be found here
    
    Args:
        data_dir (str): Specifies the directory containing the training, validation and test  data sets respectively.
        save_dir (str): Specifies the directory where the trained model will be saved. Default is the current directory.
        model_arch (str): Specifies the architecture of the model can either be 'vgg' or 'resnet'. Default is 'resnet'
        lr (float): Contains the learning rate with which to train the model. Default is 0.01.
        hidden_units (int): Contains the number of neurons in the hidden units. Default is 512.
        epoch (int): Specifies the number of epochs to train the model for. Default is 3.
        on_gpu (bool): Used to specify if training is to be done on GPU or CPU. Default is True.

    Example:
        $ python train.py "C:\\Users\\ComputerName\\Documents\\flower_data\\flower_data" --lr 0.1 --epoch 30"
        
    """
    
    
    all_dir = h.create_path(data_dir)
    train_dir = all_dir[1]
    valid_dir = all_dir[2]
    test_dir = all_dir[3]
    
    all_loaders = h.create_data_loaders(train_dir, valid_dir, test_dir)
    train_loader = all_loaders[0]
    valid_loader = all_loaders[1]
    test_loader = all_loaders[2]
    class_to_idx = all_loaders[3]
    
    # Label Mapping
    cat_to_name = h.create_cat_to_name_dict('cat_to_name.json')
    
    # Creating model
    model = h.create_model(model_arch, hidden_units)
    
    h.train(model, epoch, lr, train_loader, valid_loader, class_to_idx, cat_to_name, save_dir, on_gpu, model_arch)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A \
             simple app to train a neural net')

    parser.add_argument('data_dir', action='store',
                        help="Specifies the folder containing the training and validation data")
    parser.add_argument('--save_dir', action='store', dest='save_dir',
                        help="Specifies the folder to save the trained model")
    parser.add_argument('--arch', default='resnet', choices=['vgg', 'resnet'], dest='model_arch',
                        help="Specifies the architecture of the model. 'resnet' or 'vgg'")
    parser.add_argument('--learning_rate', action='store', type=float, default=0.01, dest='lr',
                        help="Specifies the learning rate of the model")
    parser.add_argument('--hidden_units', action='store', type=int, default=512, dest='hidden_units',
                        help="Specifies the number of hidden units in the network")
    parser.add_argument('--epochs',action='store', type=int, default=3, dest='epochs',
                        help="Specifies the number of epochs for the training")
    parser.add_argument('--gpu', action='store_false', default=True, dest='on_gpu',
                        help="Specifies if the model training will be done on GPU or CPU")

    args = parser.parse_args()
    
    print("#################....Parameters....##################")
    print("save_dir.........................{}".format(args.save_dir))
    print("arch.............................{}".format(args.model_arch))
    print("learning_rate....................{}".format(args.lr))
    print("hidden_units.....................{}".format(args.hidden_units))
    print("epochs...........................{}".format(args.epochs))
    print("gpu..............................{}".format(args.on_gpu))
    print("#################....Parameters....##################")
    
    main(args.data_dir, args.save_dir, args.model_arch, args.lr, args.hidden_units, args.epochs, args.on_gpu)
