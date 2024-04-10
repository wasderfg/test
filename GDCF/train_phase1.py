import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
from tqdm import trange
import shutil
from evaluation.evaluation_phase1 import eval_phase1_prediction
from model.GDCF import GDCF
from utils.utils import EarlyStopMonitor
from utils.data_processing import get_data

config = {
    "BJMetro": {
        "data_path": "data/BJMetro/BJMetro4.npy",
        "matrix_path": "data/BJMetro/od_matrix.npy",
        "point_path": "data/BJMetro/back_points.npy",
        "input_len": 30,
        "output_len": 30,
        "day_cycle": 1440,
        "train_day": 42,
        "val_day": 7,
        "test_day": 7,
        "day_start": 360,
        "day_end": 1320,
        "sample": 1,
        "n_nodes": 268
    },
    "NYTaxi": {
        "data_path": "data/NYTaxi/NYTaxi4.npy",
        "matrix_path": "data/NYTaxi/od_matrix.npy",
        "point_path": "data/NYTaxi/back_points.npy",
        "input_len": 1800,
        "output_len": 1800,
        "day_cycle": 86400,
        "train_day": 7,
        "val_day": 7,
        "test_day": 7,
        "day_start": 0,
        "day_end": 86400,
        "sample": 1,
        "n_nodes": 63
    }
}
### Argument and global variables
parser = argparse.ArgumentParser('GDCF training')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or BJMetro)',
                    default='NYTaxi')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=200000, help='Number of epochs')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:2", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--model', type=str, default="GDCF", help='Which model to use')

parser.add_argument('--message_dim', type=int, default=128, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=128, help='Dimensions of the memory for '
                                                               'each node')
parser.add_argument('--lambs', type=float, nargs="+", default=[1.0], help='Lamb of different time scales')


def get_embedding(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr
    MESSAGE_DIM = args.message_dim
    MEMORY_DIM = args.memory_dim

    input_len = config[DATA]["input_len"]
    output_len = config[DATA]["output_len"]
    day_cycle = config[DATA]["day_cycle"]
    day_start = config[DATA]["day_start"]
    day_end = config[DATA]["day_end"]

    Path("./output/phase1/saved_models/").mkdir(parents=True, exist_ok=True)
    Path("./output/phase1/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f'./output/phase1/saved_models/{args.data}_{args.suffix}.pth'
    get_checkpoint_path = lambda epoch: f'./output/phase1/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth'
    results_path = f"./output/phase1/results/{args.data}_{args.suffix}.pkl"
    Path("./output/phase1/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("./output/phase1/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/phase1/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    ### Extract data for training, validation and testing
    n_nodes, node_features, full_data, val_time, test_time, all_time, od_matrix_30, back_points = get_data(config[DATA])

    model = GDCF(device=device, n_nodes=n_nodes, node_features=node_features,
                 message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
                 output=output_len, lambs=args.lambs)
    model = model.to(device)

    val_losses = []
    total_epoch_times = []
    train_losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    if args.best == "":
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        num_batch = val_time // input_len
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            m_loss = []

            model.init_memory()
            model = model.train()
            batch_range = trange(num_batch)
            for j in batch_range:
                ### Training
                begin_time = j * input_len
                now_time = j * input_len + input_len
                if now_time % day_cycle < day_start or now_time % day_cycle >= day_end:
                    continue

                head, tail = back_points[begin_time // input_len], back_points[now_time // input_len]
                # [head,tail1) nowtime [tail1,tail2) nowtime+τ
                if head == tail:
                    continue

                sources_batch, destinations_batch = full_data.sources[head:tail], full_data.destinations[head:tail]
                timestamps_batch_torch = torch.Tensor(full_data.timestamps[head:tail]).to(device)
                time_diffs_batch_torch = torch.Tensor(- full_data.timestamps[head:tail] + now_time).to(device)
                edge_idxs_batch = torch.Tensor(full_data.edge_idxs[head:tail]).to(device)
                if now_time % day_cycle >= day_end:
                    predict_IND = False
                else:
                    predict_IND = True
                #分别是计算的loss和生成的embedding，只有predict_IND为true的时候才会
                loss, _ = model.compute_IND(sources_batch, destinations_batch,
                                                                    timestamps_batch_torch, now_time,
                                                                    time_diffs_batch_torch,
                                                                    edge_idxs_batch, od_matrix_30[j],
                                                                    predict_IND=predict_IND)


                if predict_IND:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    m_loss.append(loss.item())

                batch_range.set_description(f"train_loss: {m_loss[-1]} ;")

            ### Validation
            print("================================Val================================")
            val_metrics, embeddings = eval_phase1_prediction(model=model, data=full_data, back_points=back_points,
                                                          st=val_time, ed=test_time, device=device, config=config[DATA],
                                                          optimizer=optimizer, od_matrix=od_matrix_30)

            val_losses.append(val_metrics)
            train_losses.append(np.mean(m_loss))
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)

            # Save temporary results
            pickle.dump({
                "val_losses": val_losses,
                "train_losses": train_losses,
                "total_epoch_times": total_epoch_times
            }, open(results_path, "wb"))

            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
            logger.info('Epoch mean train loss: {}'.format(np.mean(m_loss)))
            logger.info('Epoch val loss: loss: {}'.format(val_metrics))
            # Early stopping
            ifstop, ifimprove = early_stopper.early_stop_check(val_metrics)
            if ifstop:
                logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                break
            else:
                torch.save(
                    {"statedict": model.state_dict(), "memory": model.backup_memory()}, get_checkpoint_path(epoch))
        logger.info('Saving GDCF model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('GDCF model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    model.load_state_dict(best_model_param["statedict"])
    model.restore_memory(best_model_param["memory"])
    model.init_memory()
    # Test
    print("================================Test================================")
    test_metrics, embeddings = eval_phase1_prediction(model=model, data=full_data,
                                                   back_points=back_points, st=0, ed=all_time,
                                                   device=device, config=config[DATA], optimizer=optimizer,
                                                   od_matrix=od_matrix_30)

    logger.info(
        'Test statistics:-- loss: {}'.format(test_metrics))
    # Save results for this run
    pickle.dump({
        "val_losses": val_losses,
        "test_metrics": test_metrics,
        "train_losses": train_losses,
        "total_epoch_times": total_epoch_times,
        "embeddings": embeddings
    }, open(results_path, "wb"))


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    get_embedding(args)
