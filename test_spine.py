import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import argparse
from utils.preprocessing import scale2actual
from models.dense_net_regression import DenseNetRegression
from data_providers.utils import get_data_provider_by_name
import numpy as np
import random
from utils.evaluate import mae, mse, pearson
import os
from run_evaluate import call_evaluate
import tensorflow as tf
from scipy import io as sio

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

    dataDir = args.data_path
    for folderInd in range(1,6):
        args.folderInd = folderInd
        args.save_path = os.sep.join([args.data_path, 'out', 'folder' + str(folderInd), 'model'])
        args.logs_path = os.sep.join([args.data_path, 'out', 'folder' + str(folderInd), 'logs'])
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
            train_params['folderInd'] = folderInd
            train_params['data_path'] = args.data_path
            train_params['knn'] = args.knn
        print("Train params:")
        for k, v in train_params.items():
            print("\t%s: %s" % (k, v))

        print("Prepare training data...")

        data_provider = get_data_provider_by_name(args.dataset, train_params)
        batch_size = train_params['batch_size']
        print("Initialize the model..")
        tf.reset_default_graph()
        model = DenseNetRegression(data_provider=data_provider, **model_params)

        outDir = os.sep.join([dataDir, 'out', 'results', model.model_identifier])
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
                os.makedirs(outDir)
            # for training dataset

            # loss = model.test(data_provider.train, batch_size=43)
            # print(loss)

            train_pre_y = model.predict(data_provider.train, batch_size=batch_size)
            train_embedding = model.get_embedding(data_provider.train, batch_size=batch_size)
            train_y = data_provider.train.labels

            act_train_pre_y = scale2actual(train_pre_y, pixelSizes[trainInd, :], np.tile(height, train_pre_y.shape),
                                         np.tile(width, train_pre_y.shape), mode='height')
            act_train_y = scale2actual(train_y, pixelSizes[trainInd, :], np.tile(height, train_y.shape),
                                     np.tile(width, train_y.shape), mode='height')
            train_pearson = pearson(act_train_pre_y, act_train_y)
            train_mae = mae(act_train_pre_y, act_train_y)
            train_mse = mse(act_train_pre_y, act_train_y)
            train_std = np.std(train_pre_y,axis=0)

            # for validation dataset
            val_pre_y = model.predict(data_provider.test, batch_size=batch_size)
            val_embedding = model.get_embedding(data_provider.test, batch_size=batch_size)
            val_y = data_provider.test.labels
            act_val_pre_y = scale2actual(val_pre_y, data_provider.pixelSizes[valInd,:], np.tile(height, val_pre_y.shape),
                                     np.tile(width, val_pre_y.shape), mode='height')
            act_val_y = scale2actual(val_y, data_provider.pixelSizes[valInd,:], np.tile(height, val_y.shape),
                                     np.tile(width, val_y.shape), mode='height')
            val_pearson = pearson(act_val_pre_y, act_val_y)
            val_mae = mae(act_val_pre_y, act_val_y)
            val_mse = mse(act_val_pre_y, act_val_y)
            val_std = np.std(val_pre_y,axis=0)

            val_mae_err = np.mean(np.abs(act_val_pre_y - act_val_y), axis=1)
            print('folder: {} {} mean train mae= {}  mean train pearson= {} mean train mse= {}'.format(
                str(folderInd), 'out', str(np.mean(train_mae)), str(np.mean(train_pearson)), str(np.mean(train_mse))))
            print('folder: {} {} mean val mae= {}  mean val pearson= {} mean val mse= {}'.format(
                str(folderInd), 'out', str(np.mean(val_mae)), str(np.mean(val_pearson)), str(np.mean(val_mse))))
            print('folder: {} train_std = {}'.format(str(folderInd), str(train_std)))
            print('folder: {} val_std = {}'.format(str(folderInd), str(val_std)))

            np.savez('/'.join([outDir, 'predict-fold' + str(folderInd) + '.npz']),
                     train_pre_y=train_pre_y, val_pre_y=val_pre_y,
                     train_mae=train_mae,
                     train_mse=train_mse,
                     train_pearson=train_pearson,
                     val_mae=val_mae,
                     val_mse=val_mse,
                     val_pearson=val_pearson,
                     val_mae_err=val_mae_err,
                     train_y=train_y, val_y=val_y,
                     pixelSizes=pixelSizes, height=height, width=width)

            sio.savemat(os.sep.join([outDir, 'embedding-fold' + str(folderInd) + '.mat']), {'train_embedding':train_embedding, 'train_y':train_y, 'val_embedding':val_embedding, 'val_y':val_y})
    total_train_mae, total_train_mse, total_train_pearson, total_val_mae, total_val_mse, total_val_pearson = call_evaluate(dataDir, 'out', model.model_identifier)

    np.savez('/'.join([dataDir, 'out', 'results', model.model_identifier, 'evaluation.npz']), val_pearson=total_val_pearson,
             val_mae=total_val_mae,
             val_mse=total_val_mse, train_pearson=total_train_pearson, train_mae=total_train_mae, train_mse=total_train_mse)

    print('{} mean train mae= {}  mean train pearson= {} mean train mse= {}'.format(
        'out', str(np.mean(total_train_mae)), str(np.mean(total_train_pearson)), str(np.mean(total_train_mse))))
    print('{} mean val mae= {}  mean val pearson= {} mean val mse= {}'.format(
        'out', str(np.mean(total_val_mae)), str(np.mean(total_val_pearson)), str(np.mean(total_val_mse))))
