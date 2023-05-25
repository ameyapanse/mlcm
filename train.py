import argparse
import datetime
import os
import time
import torch

from model import encoder
import pickle
import json
import datautils
from utils import init_dl_program, name_with_datetime
import numpy as np


def save_checkpoint_callback(
        save_every=1,
        unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="PAMAP2" , help='The dataset name')
    parser.add_argument('--run_name', default='default_run',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--objective', default='gzsl', help='Data Folder')
    parser.add_argument('--fold', default=0, type=int,help='Data Fold')
    parser.add_argument('--axis', default=None, type=int, help='Data Axis if single variable treatment')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=100, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('-h1_dims', type=int, default=300, help='The size of first layer')
    parser.add_argument('-h2_dims', type=int, default=150, help='The size of second layer')
    parser.add_argument('-h3_dims', type=int, default=150, help='The size of third layer')
    parser.add_argument('-h4_dims', type=int, default=768, help='The size of embedding layer.Should match the size of label embeddings')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=30, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')

    args = parser.parse_args()
    args.dataset = 'PAMAP2'
    args.run_name = 'default'
    args.eval=True
    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.dataset == 'PAMAP2':
        fold_path = args.objective + '/fold' + str(args.fold)
        if args.fold == 0:
            fold_path=None
        train_data, train_labels, train_embeddings,\
            test_data, test_labels, test_embeddings,\
            label_embeddings \
            = datautils.load_PAMAP2(fold_path, args.axis)
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        input_dims=train_data.shape[-1],
        # h1_dims=args.h1_dims,
        # h2_dims=args.h2_dims,
        # h3_dims=args.h3_dims,
        # h4_dims=args.h4_dims,
        label_embeddings=label_embeddings,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    if not fold_path:
        fold_path='/fold0'
    run_dir = 'training/' + args.run_name + '_' + args.dataset + '/' + fold_path
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = encoder(
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        **config
    )
    # model.to(cuda)
    loss_log = model.fit(
        train_data,
        train_labels,
        train_embeddings,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )

    #model.save(f'{run_dir}/model.pkl')
    torch.save(model.state_dict(), f'{run_dir}/model.pkl')
    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        model.eval()
        preds = np.array(model.predict(train_data))
        train_acc = np.sum(preds == train_labels)/ preds.shape[0]
        print('Train Accuracy : ', train_acc )

        preds = np.array(model.predict(test_data))
        test_acc = np.sum(preds == test_labels)/ preds.shape[0]
        print('Test Accuracy : ', test_acc)

        results = dict()
        results['arguments'] = vars(args)
        results['loss'] = loss_log
        results['train_acc']= train_acc
        results['test_acc'] = test_acc
        with open(f'{run_dir}/results.pkl', 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
        res_str = json.dumps(results, indent=1)
        with open(f'{run_dir}/results.txt', 'w') as file:
            file.write(res_str)



    print("Finished.")