import argparse
import datetime
import os
import time

from model import encoder

import datautils
from utils import init_dl_program, name_with_datetime


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
    parser.add_argument('dataset', default="PAMAP2" , help='The dataset name')
    parser.add_argument('run_name',
                        help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--gpu', type=int, default=0,
                        help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('-h1_dims', type=int, default=100, help='The size of first layer')
    parser.add_argument('-h2_dims', type=int, default=100, help='The size of second layer')
    parser.add_argument('-h3_dims', type=int, default=100, help='The size of third layer')
    parser.add_argument('-h4_dims', type=int, default=100, help='The size of embedding layer.Should match the size of label embeddings')
    parser.add_argument('--max-train-length', type=int, default=3000,
                        help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None,
                        help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None,
                        help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')
    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    print('Loading data... ', end='')
    if args.dataset == 'PAMAP2':
        train_data, train_labels, test_data, test_labels, embeddings = datautils.load_PAMAP2()
    else:
        raise ValueError(f"Unknown loader {args.loader}.")
    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        input_dims=train_data.shape[-1],
        h1_dims=args.h1_dims,
        h2_dims=args.h2_dims,
        h3_dims=args.h3_dims,
        h4_dims=args.h4_dims,
        label_embeddings=embeddings,
        max_train_length=args.max_train_length
    )

    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'training/' + args.dataset + '__' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)

    t = time.time()

    model = encoder(
        device=device,
        **config
    )
    model.to(device)
    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )

    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")
    print("Finished.")