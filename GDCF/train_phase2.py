import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random
import tqdm
import shutil
from utils.utils import EarlyStopMonitor
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error

config = {
    "BJMetro": {
        "data_path": "output/phase1/results/BJMetro_.pkl",
        "matrix_path": "data/BJMetro/od_matrix.npy",
        "day_cycle": 48,
        "train_day": 42,
        "val_day": 7,
        "test_day": 7,
        "day_start": 12,
        "day_end": 44,
        "n_nodes": 268,
        "batch_size": 32
    },
    "NYTaxi": {
        "data_path": "output/phase1/results/NYTaxi_.pkl",
        "matrix_path": "data/NYTaxi/od_matrix.npy",
        "day_cycle": 48,
        "train_day": 7,
        "val_day": 7,
        "test_day": 7,
        "day_start": 0,
        "day_end": 48,
        "n_nodes": 63,
        "batch_size": 64
    }
}

### Argument and global variables
parser = argparse.ArgumentParser('GDCF training')
parser.add_argument('--data', type=str, help='Dataset name (eg. NYTaxi or BJMetro)',
                    default='NYTaxi')
parser.add_argument('--emd_path', type=str, help='Embedding Path',
                    default='')
parser.add_argument('--seed', type=int, default=0, help='Random seed')
parser.add_argument('--suffix', type=str, default='', help='Suffix to name the checkpoints')
parser.add_argument('--best', type=str, default='', help='Path of the best model')
parser.add_argument('--n_epoch', type=int, default=10000, help='Number of epochs')
parser.add_argument('--few_day', type=int, default=139, help='Number of training days')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=50, help='Patience for early stopping')
parser.add_argument('--device', type=str, default="cuda:1", help='Idx for the gpu to use: cpu, cuda:0, etc.')
parser.add_argument('--task', type=str, default="od", help='What task: od, i, o')
parser.add_argument('--loss', type=str, default="mse", help='Loss function')


class PredictionLayer(nn.Module):
    def __init__(self, embedding_dim, n_nodes):
        super(PredictionLayer, self).__init__()
        self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, embeddings):
        B, N, F = embeddings.shape
        return self.w(torch.cat(
            [embeddings.repeat([1, 1, self.n_nodes]).reshape([B, self.n_nodes * self.n_nodes, -1]),
             embeddings.repeat([1, self.n_nodes, 1])],
            dim=2)).reshape([B, self.n_nodes, self.n_nodes])


class PredictionLayer_s(nn.Module):
    def __init__(self, embedding_dim):
        super(PredictionLayer_s, self).__init__()
        # self.n_nodes = n_nodes
        self.w = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim, int(embedding_dim / 2)),
            nn.LeakyReLU(),
            nn.Linear(int(embedding_dim / 2), 1),
        )

    def forward(self, embeddings):
        B, N, F = embeddings.shape
        return self.w(embeddings).reshape([B, N])


def get_data(config):
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    train_day = config["train_day"]
    val_day = config["val_day"]
    test_day = config["test_day"]
    embeddings = pickle.load(open(config["data_path"], "rb"))
    embeddings = embeddings["embeddings"]
    full_steps, n_nodes, embedding_dim = embeddings.shape
    od_matrix = np.load(config["matrix_path"]).reshape([-1, day_cycle, n_nodes, n_nodes])[:, day_start:day_end]
    od_matrix = od_matrix.reshape([-1, n_nodes, n_nodes])
    if day_start == 0:
        offset = 1
        od_matrix = od_matrix[1:]
    else:
        offset = 0
    full_set = np.arange(full_steps + offset)
    train_set = full_set[offset:train_day * (day_end - day_start)] - offset
    val_set = full_set[train_day * (day_end - day_start):(train_day + val_day) * (day_end - day_start)] - offset
    test_set = full_set[(train_day + val_day) * (day_end - day_start):] - offset
    data_loaders = {"train": DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], drop_last=False),
                    "val": DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], drop_last=False),
                    "test": DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], drop_last=False)}

    return n_nodes, embedding_dim, od_matrix, embeddings, data_loaders


def get_data_few(config, few_day):
    day_cycle = config["day_cycle"]
    day_start = config["day_start"]
    day_end = config["day_end"]
    train_day = config["train_day"]
    val_day = config["val_day"]
    test_day = config["test_day"]
    embeddings = pickle.load(open(config["data_path"], "rb"))
    embeddings = embeddings["embeddings"]
    full_steps, n_nodes, embedding_dim = embeddings.shape
    od_matrix = np.load(config["matrix_path"]).reshape([-1, day_cycle, n_nodes, n_nodes])[:, day_start:day_end]
    od_matrix = od_matrix.reshape([-1, n_nodes, n_nodes])
    if day_start == 0:
        offset = 1
        od_matrix = od_matrix[1:]
    else:
        offset = 0
    full_set = np.arange(full_steps + offset)
    train_set = full_set[
                offset + (train_day - few_day) * (day_end - day_start):train_day * (day_end - day_start)] - offset
    val_set = full_set[train_day * (day_end - day_start):(train_day + val_day) * (day_end - day_start)] - offset
    test_set = full_set[(train_day + val_day) * (day_end - day_start):] - offset
    data_loaders = {"train": DataLoader(train_set, shuffle=True, batch_size=config["batch_size"], drop_last=False),
                    "val": DataLoader(val_set, shuffle=False, batch_size=config["batch_size"], drop_last=False),
                    "test": DataLoader(test_set, shuffle=False, batch_size=config["batch_size"], drop_last=False)}

    return n_nodes, embedding_dim, od_matrix, embeddings, data_loaders


def calculate_metrics(stacked_prediction, stacked_label):
    stacked_prediction[stacked_prediction < 0] = 0
    reshaped_prediction = stacked_prediction.reshape(-1)
    reshaped_label = stacked_label.reshape(-1)
    mse = mean_squared_error(reshaped_prediction, reshaped_label)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(reshaped_prediction, reshaped_label)
    pcc = np.corrcoef(reshaped_prediction, reshaped_label)[0][1]
    smape = np.mean(2 * np.abs(reshaped_prediction - reshaped_label) / (
            np.abs(reshaped_prediction) + np.abs(reshaped_label) + 1))
    return (mse, rmse, mae, pcc, smape)


def train_phase2(args):
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    NUM_EPOCH = args.n_epoch
    device = args.device
    DATA = args.data
    LEARNING_RATE = args.lr

    Path(f"./output/{args.task}/saved_models/").mkdir(parents=True, exist_ok=True)
    Path(f"./output/{args.task}/saved_checkpoints/").mkdir(parents=True, exist_ok=True)
    MODEL_SAVE_PATH = f"./output/{args.task}/saved_models/{args.data}_{args.suffix}.pth"
    get_checkpoint_path = lambda epoch: f"./output/{args.task}/saved_checkpoints/{args.data}_{args.suffix}_{epoch}.pth"
    results_path = f"./output/{args.task}/results/{args.data}_{args.suffix}.pkl"
    Path(f"./output/{args.task}/results/").mkdir(parents=True, exist_ok=True)

    ### set up logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path(f"./output/{args.task}/log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(f"./output/{args.task}/log/{str(time.time())}_{args.data}_{args.suffix}.log")
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)

    if args.emd_path != "":
        print("assigned embedding")
        config[DATA]["data_path"] = args.emd_path
    ### Extract data for training, validation and testing
    if "few" in args.suffix:
        n_nodes, embedding_dim, od_matrix, embeddings, data_loaders = get_data_few(config[DATA], args.few_day)
    else:
        n_nodes, embedding_dim, od_matrix, embeddings, data_loaders = get_data(config[DATA])
    print("embeddings",embeddings.shape) #(1007, 63, 128)
    if args.task == "od":
        model = PredictionLayer(embedding_dim=embedding_dim, n_nodes=n_nodes)
    elif args.task in ["i", "o"]:
        model = PredictionLayer_s(embedding_dim=embedding_dim)
    else:
        raise NotImplementedError

    if args.loss == "mse":
        criterion = torch.nn.MSELoss()
    else:
        raise NotImplementedError

    model = model.to(device)

    val_mses = []
    epoch_times = []
    total_epoch_times = []
    train_mses = []
    if args.best == "":
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        early_stopper = EarlyStopMonitor(max_round=args.patience, higher_better=False)
        ifstop = False
        for epoch in range(NUM_EPOCH):
            print("================================Epoch: %d================================" % epoch)
            start_epoch = time.time()
            logger.info('start {} epoch'.format(epoch))
            for phase in ["train", "val"]:
                label, prediction = [], []
                if phase == "train":
                    model = model.train()
                else:
                    model = model.eval()
                batch_range = tqdm.tqdm(data_loaders[phase])
                for ind in batch_range:
                    # Predict OD, get updated memories and messages
                    batch_embeddings = torch.Tensor(embeddings[ind]).to(device)
                    od_matrix_real = od_matrix[ind]
                    if args.task == "od":
                        real_data = od_matrix_real
                    elif args.task == "i":
                        real_data = np.sum(od_matrix_real, axis=1)
                    elif args.task == "o":
                        real_data = np.sum(od_matrix_real, axis=2)
                    else:
                        raise NotImplementedError
                    predicted_data = model(batch_embeddings)
                    if phase == "train":
                        optimizer.zero_grad()
                        loss = criterion(predicted_data, torch.Tensor(real_data).to(device))
                        loss.backward()
                        optimizer.step()
                        # m_loss.append(loss.item())
                        batch_range.set_description(f"train_loss: {loss.item()};")
                    label.append(real_data)
                    prediction.append(predicted_data.cpu().detach().numpy())
                concated_label = np.concatenate(label)
                concated_prediction = np.concatenate(prediction)
                metrics = calculate_metrics(concated_prediction, concated_label)
                logger.info(
                    'Epoch {} {} metric: mse, rmse, mae, pcc, smape, {}, {}, {}, {}, {}'.format(epoch, phase, *metrics))
                if phase == "train":
                    train_mses.append(metrics[0])
                elif phase == "val":
                    val_mses.append(metrics[0])
                    # Early stopping
                    ifstop, ifimprove = early_stopper.early_stop_check(metrics[0])
                    if ifstop:
                        logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
                        logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
                        logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
                    else:
                        logger.info('No improvement over {} epochs'.format(early_stopper.num_round))
                        torch.save(
                            {"statedict": model.state_dict()},
                            get_checkpoint_path(epoch))
            if ifstop:
                break
            total_epoch_time = time.time() - start_epoch
            total_epoch_times.append(total_epoch_time)
            logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))

        # Save temporary results
        pickle.dump({
            "val_mses": val_mses,
            "train_losses": train_mses,
            "epoch_times": epoch_times,
            "total_epoch_times": total_epoch_times
        }, open(results_path, "wb"))

        logger.info('Saving GDCF model')
        shutil.copy(get_checkpoint_path(early_stopper.best_epoch), MODEL_SAVE_PATH)
        logger.info('GDCF model saved')
        best_model_param = torch.load(get_checkpoint_path(early_stopper.best_epoch))
    else:
        best_model_param = torch.load(args.best)

    # load model parameters, memories from best epoch on val dataset
    model.load_state_dict(best_model_param["statedict"])
    # Test
    print("================================Test================================")

    model = model.eval()
    batch_range = tqdm.tqdm(data_loaders["test"])
    label, prediction = [], []
    for ind in batch_range:
        batch_embeddings = torch.Tensor(embeddings[ind]).to(device)
        predicted_data = model(batch_embeddings)
        od_matrix_real = od_matrix[ind]
        if args.task == "od":
            real_data = od_matrix_real
        elif args.task == "i":
            real_data = np.sum(od_matrix_real, axis=1)
        elif args.task == "o":
            real_data = np.sum(od_matrix_real, axis=2)
        else:
            raise NotImplementedError
        label.append(real_data)
        prediction.append(predicted_data.cpu().detach().numpy())
    concated_label = np.concatenate(label)
    concated_prediction = np.concatenate(prediction)
    test_metrics = calculate_metrics(concated_prediction, concated_label)

    logger.info(
        'Test statistics:-- mse: {}, rmse: {}, mae: {}, pcc: {}, smape:{}'.format(*test_metrics))
    # Save results for this run
    pickle.dump({
        "val_mses": val_mses,
        "test_mse": test_metrics[0],
        "test_rmse": test_metrics[1],
        "test_mae": test_metrics[2],
        "test_pcc": test_metrics[3],
        "test_smape": test_metrics[4],
        "epoch_times": epoch_times,
        "train_losses": train_mses,
        "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))


if __name__ == '__main__':
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    train_phase2(args)
