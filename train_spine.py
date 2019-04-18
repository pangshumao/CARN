import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
import random
from models.dense_net_regression import DenseNetRegression
from data_providers.utils import get_data_provider_by_name
import numpy as np

train_params_spine = {
    'batch_size': 8,
    'n_epochs': 500,
    # 'initial_learning_rate': 0.1,
    'reduce_lr_epoch_1': 150,  # epochs * 1/3
    'reduce_lr_epoch_2': 400,  # epochs * 2/3
    'validation_set': True,
    'validation_split': None,  # None or float
    'shuffle': 'every_epoch',  # None, once_prior_train, every_epoch
    'normalization': 'by_chanels',  # 'None', divide_256, divide_255, by_chanels, divide_mean, subtract_mean
}

def get_train_params_by_name(name):
    if name in ['SPINE', 'SPINE+']:
        return train_params_spine

if __name__ == '__main__':
    random.seed(10)
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
        '--model_type', '-m', type=str, choices=['DenseNet', 'DenseNet-BC',
                                                  'CARN', 'CNN', 'CARN-LSPMR', 'CNN-LSPMR'],
        default='CARN-LSPMR',
        help='What type of model to use, DenseNet, DenseNet-BC, CARN, CNN, CARN-LSPMR, or CNN-LSPMR')
    parser.add_argument(
        '--data_path', '-dp', type=str, metavar='',default='I:\\ImageData\\spine\\allData\\FinalCARN',
        help='set the data path')
    parser.add_argument(
        '--folderInd', '-fi', type=int, metavar='',default=1,
        help='the folder index, range from 1 to 5')
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
        default=6,
        help='The number of AU')
    parser.add_argument(
        '--knn', type=int, metavar='',
        help='the k-nearest neighbor in LAE for ALSCMR',
        default=20)
    parser.add_argument(
        '--alscmrWeight', type=float, metavar='',
        help='the loss weight of LAE',
        default=101.0)
    parser.add_argument(
        '--lspmrWeight', type=float, metavar='',
        help='the sdne loss weight',
        default=0.005)
    parser.add_argument(
        '--dataset', '-ds', type=str,
        choices=['SPINE', 'SPINE+'],
        default='SPINE+',
        help='What dataset should be used')
    parser.add_argument(
        '--total_blocks', '-tb', type=int, default=4, metavar='',
        help='Total blocks of layers stack (default: %(default)s)')
    parser.add_argument(
        '--keep_prob', '-kp', type=float, metavar='',default=1.0,
        help="Keep probability for dropout.")
    parser.add_argument(
        '--weight_decay', '-wd', type=float, default=5e-3, metavar='',
        help='Weight decay for optimizer (default: %(default)s)')
    parser.add_argument(
        '--nesterov_momentum', '-nm', type=float, default=0.9, metavar='',
        help='Nesterov momentum (default: %(default)s)')
    parser.add_argument(
        '--reduction', '-red', type=float, default=1.0, metavar='',
        help='reduction Theta at transition layer for DenseNets-BC models')

    parser.add_argument(
        '--logs', dest='should_save_logs', action='store_true',
        help='Write tensorflow logs')
    parser.add_argument(
        '--no-logs', dest='should_save_logs', action='store_false',
        help='Do not write tensorflow logs')
    parser.set_defaults(should_save_logs=True)
    parser.add_argument(
        '--saves', dest='should_save_model', action='store_true',
        help='Save model during training')
    parser.add_argument(
        '--no-saves', dest='should_save_model', action='store_false',
        help='Do not save model during training')
    parser.set_defaults(should_save_model=True)
    parser.add_argument(
        '--renew-logs', dest='renew_logs', action='store_true', default=False,
        help='Erase previous logs for model if exists.')
    parser.add_argument(
        '--not-renew-logs', dest='renew_logs', action='store_false',
        help='Do not erase previous logs for model if exists.')
    parser.set_defaults(renew_logs=True)
    parser.add_argument(
        '--reuse_model', type=bool, metavar='', default=False,
        help='whether reuse the existed model, True or False')

    args = parser.parse_args()

    if not args.keep_prob:
        if args.dataset in ['SPINE', 'SPINE+']:
            args.keep_prob = 0.8
        else:
            args.keep_prob = 1.0
    if args.model_type == 'DenseNet':
        args.carn_mode = False
        args.cnn_mode = False
        args.bc_mode = False
        args.reduction = 1.0
        args.lspmr_mode = False
    elif args.model_type == 'DenseNet-BC':
        args.carn_mode = False
        args.cnn_mode = False
        args.bc_mode = True
        args.lspmr_mode = False
    elif args.model_type == 'CARN':
        args.carn_mode = True
        args.cnn_mode = False
        args.bc_mode = False
        args.lspmr_mode = False
    elif args.model_type == 'CNN':
        args.carn_mode = False
        args.cnn_mode = True
        args.bc_mode = False
        args.lspmr_mode = False
    elif args.model_type == 'CARN-LSPMR':
        args.carn_mode = True
        args.cnn_mode = False
        args.bc_mode = False
        args.lspmr_mode = True
    elif args.model_type == 'CNN-LSPMR':
        args.carn_mode = False
        args.cnn_mode = True
        args.bc_mode = False
        args.lspmr_mode = True

    model_params = vars(args)

    if not args.train and not args.test:
        print("You should train or test your network. Please check params.")
        exit()

    # some default params dataset/architecture related
    train_params = get_train_params_by_name(args.dataset)
    print("Params:")
    for k, v in model_params.items():
        print("\t%s: %s" % (k, v))
    print("Train params:")
    for k, v in train_params.items():
        print("\t%s: %s" % (k, v))

    print("Prepare training data...")
    train_params['data_path'] = args.data_path
    if args.dataset in ['SPINE', 'SPINE+']:
        train_params['folderInd'] = args.folderInd
        train_params['knn'] = args.knn
    args.save_path = os.sep.join([args.data_path, 'out', 'folder' + str(args.folderInd), 'model'])
    args.logs_path = os.sep.join([args.data_path, 'out', 'folder' + str(args.folderInd), 'logs'])
    if not os.path.exists(os.sep.join([args.data_path, 'out'])):
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.logs_path, exist_ok=True)
    elif not os.path.exists(os.sep.join([args.data_path, 'out', 'folder' + str(args.folderInd)])):
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.logs_path, exist_ok=True)
    elif not os.path.exists(args.save_path):
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(args.logs_path, exist_ok=True)

    data_provider = get_data_provider_by_name(args.dataset, train_params)
    print("Initialize the model..")
    model = DenseNetRegression(data_provider=data_provider, **model_params)
    if args.train:
        print("Data provider train images: ", data_provider.train.num_examples)
        if args.reuse_model:
            model.load_model()
        model.train_all_epochs(train_params)
        model.train_summary_writer.close()
        model.test_summary_writer.close()
    if args.test:
        if not args.train:
            model.load_model()
        print("Data provider test images: ", data_provider.test.num_examples)
        print("Testing...")
        if model.lspmr_mode:
            mae_loss, lspmr_loss = model.test(data_provider.test, batch_size=train_params['batch_size'])
            print("mean mae: {}, mean lspmr_loss: {}".format(mae_loss, lspmr_loss))
        else:
            loss = model.test(data_provider.test, batch_size=train_params['batch_size'])
            print("mean mae: %f" % loss)
