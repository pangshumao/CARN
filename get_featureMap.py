import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
from utils.preprocessing import scale2actual
from models.dense_net_regression import DenseNetRegression
from data_providers.utils import get_data_provider_by_name
import numpy as np
from utils.evaluate import mae, mse, pearson
import os
from run_evaluate import call_evaluate
import tensorflow as tf
from scipy import io as sio

train_params_cifar = {
    'batch_size': 64,
    'n_epochs': 300,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 0.5
    'reduce_lr_epoch_2': 225,  # epochs * 0.75
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_spine = {
    'batch_size': 8,
    'n_epochs': 500,
    # 'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 1/3
    'reduce_lr_epoch_2': 400,  # epochs * 2/3
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # None, divide_256, divide_255, by_chanels
}

train_params_svhn = {
    'batch_size': 64,
    'n_epochs': 40,
    'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 20,
    'reduce_lr_epoch_2': 30,
    'validation_set': True,
    'validation_split': None,  # you may set it 6000 as in the paper
    'shuffle': True,  # shuffle dataset every epoch or not
    'normalization': 'divide_255',
}

def get_train_params_by_name(name):
    if name in ['C10', 'C10+', 'C100', 'C100+']:
        return train_params_cifar
    if name == 'SVHN':
        return train_params_svhn
    if name in ['SPINE', 'SPINE+']:
        return train_params_spine


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--train', action='store_true',
        help='Train the model')
    parser.add_argument(
        '--test', action='store_true',
        help='Test model for required dataset if pretrained model exists.'
             'If provided together with `--train` flag testing will be'
             'performed right after training.')
    parser.add_argument(
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC', 'DenseNet-BC-Gate',
                                                 'DenseNet-Gan', 'DenseNet-BC-Gan', 'DenseNet-BC-Gate-Gan', 'Vgg',
                                                 'Vgg-Gan', 'GCNN', 'GCNN-Gan', 'CNN', 'GCNN-SDNE', 'CNN-SDNE'],
        default='GCNN-SDNE',
        help='What type of model to use, DenseNet, DenseNet-BC, DenseNet-Gate, DenseNet-BC-Gate, '
             'DenseNet-Gan, DenseNet-BC-Gan, DenseNet-BC-Gate-Gan, Vgg or Vgg-Gan')
    parser.add_argument(
        '--gamma', '-g', type=float,
        default=0.5,
        help='the weight of real loss in balance'
    )
    parser.add_argument(
        '--lambda_g', type=float,
        default=0.0,
        help='the weight of mae loss'
    )
    parser.add_argument(
        '--lr', type=float,
        default=0.04,
        help='the initial learning rate'
    )
    parser.add_argument(
        '--growth_rate', '-k', type=int,
        default=48,
        help='Grows rate for every layer, '
             'choices were restricted to used in paper')
    parser.add_argument(
        '--depth', '-d', type=int,
        default=8,
        help='Depth of whole network, restricted to paper choices')
    parser.add_argument(
        '--knn', type=int, metavar='',
        help='the k-nearest neighbor in LAE',
        default=20)
    parser.add_argument(
        '--laeWeight', type=float, metavar='',
        help='the loss weight of LAE',
        default=101.0)
    parser.add_argument(
        '--sdneWeight', type=float, metavar='',
        help='the sdne loss weight',
        default=0.005)
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['C10', 'C10+', 'C100', 'C100+', 'SVHN', 'SPINE', 'SPINE+'],
        default='SPINE+',
        help='What dataset should be used')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=4, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='', default=0.5,
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=5e-3, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=0.5, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')
    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)
    parser.add_argument(
        '--logs_path', '-lp', type=str, default='logs', metavar='',
        help='set the path of the logs')
    parser.add_argument(
        '--y_mode', '-ym', type=str, default='disc_height', metavar='',
        help='disc_height or vb_height')
    parser.add_argument(
        '--folderInd', '-fi', type=int, metavar='',
        help='the folder index, range from 1 to 5')
    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)
    parser.add_argument(
        '--save_path', '-sp', type=str, metavar='',
        help='set the path of the saved model')
    parser.add_argument(
        '--data_path', '-dp', type=str, metavar='',
        help='set the data path')
    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true',
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)
    parser.add_argument(
        '--reuse_model', type=bool, metavar='', default=True,
        help='whether reuse the existed model, True or False')

    args = parser.parse_args()

    if not args.keep_prob:
        if args.dataset in ['C10', 'C100', 'SVHN', 'SPINE', 'SPINE+']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = False
        args.reduction = 1.0
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-BC':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-Gate':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = False
        args.reduction = 1.0
        args.gate_mode = True
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-BC-Gate':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = True
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-Gan':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = False
        args.reduction = 1.0
        args.gate_mode = False
        args.gan_mode = True
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-BC-Gan':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = False
        args.gan_mode = True
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'DenseNet-BC-Gate-Gan':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = True
        args.gan_mode = True
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'Vgg':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = True
        args.sdne_mode = False
    elif args.model_type == 'Vgg-Gan':
        args.gcnn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.gate_mode = False
        args.gan_mode = True
        args.vgg_mode = True
        args.sdne_mode = False
    elif args.model_type == 'GCNN':
        args.gcnn_mode = True
        args.cnn_mode = False
        args.bc_mode = False
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'GCNN-Gan':
        args.gcnn_mode = True
        args.cnn_mode = False
        args.bc_mode = False
        args.gate_mode = False
        args.gan_mode = True
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'CNN':
        args.gcnn_mode = False
        args.cnn_mode = True
        args.bc_mode = False
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = False
    elif args.model_type == 'GCNN-SDNE':
        args.gcnn_mode = True
        args.cnn_mode = False
        args.bc_mode = False
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = True
    elif args.model_type == 'CNN-SDNE':
        args.gcnn_mode = False
        args.cnn_mode = True
        args.bc_mode = False
        args.gate_mode = False
        args.gan_mode = False
        args.vgg_mode = False
        args.sdne_mode = True

    y_mode = args.y_mode
    dataDir = args.data_path
    for folderInd in range(1,6):
        args.save_path = os.sep.join([args.data_path, y_mode, 'folder' + str(folderInd), 'model'])
        model_params = vars(args)

        if not args.train and not args.test:
            print("You should train or test your network. Please check params.")
            exit()

        # some default params dataset/architecture related
        train_params = get_train_params_by_name(args.dataset)
        print("Params:")
        for k, v in model_params.items():
            print("\t%s: %s" % (k, v))
        if args.dataset == 'SPINE' or args.dataset == 'SPINE+':
            train_params['y_mode'] = args.y_mode
            train_params['folderInd'] = folderInd
            train_params['save_path'] = args.data_path
            train_params['knn'] = args.knn
        print("Train params:")
        for k, v in train_params.items():
            print("\t%s: %s" % (k, v))

        print("Prepare training data...")

        data_provider = get_data_provider_by_name(args.dataset, train_params)
        print("Initialize the model..")
        tf.reset_default_graph()
        model = DenseNetRegression(data_provider=data_provider, **model_params)

        outDir = os.sep.join([dataDir, y_mode, 'results', model.model_identifier])
        if args.test:
            if not args.train:
                model.load_model()
            print("Data provider test images: ", data_provider.test.num_examples)
            print("Testing...")
            pixelSizes = data_provider.pixelSizes
            height = data_provider.height
            width = data_provider.width
            trainInd = data_provider.trainInd
            valInd = data_provider.valInd
            if not os.path.exists(outDir):
                os.mkdir(outDir)

            images = data_provider.test.images
            initial_conv_1 = model.get_feature_map(images, 'Initial_convolution_1')
            for i in range(1,7):
                gate = model.get_feature_map(images, 'gate_' + str(i))
                selected_fea = model.get_feature_map(images, 'selected_fea_' + str(i))
                sio.savemat(os.sep.join([outDir, 'featureMap_' + str(i) + '_fold' + str(folderInd) + '.mat']), {'images':images, 'initial_conv_1':initial_conv_1, 'gate':gate, 'selected_fea':selected_fea})

