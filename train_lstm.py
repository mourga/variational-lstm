import argparse
import os

import sys

import torch
from torch import nn
from torch.utils.data import DataLoader

from modules.data.collates import ClfCollate
from modules.data.datasets import ClfDataset
from modules.data.samplers import BucketBatchSampler
from modules.models import RNNClassifier
from modules.trainers.trainers import ClfTrainer
from sys_config import EMB_DIR
from utils.data_parsing import load_dataset
from utils.early_stopping import EarlyStopping
from utils.general import number_h
from utils.opts import train_options
from utils.training import acc, precision_macro, recall_macro, f1_macro

sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))



def train_lstm_model(X_train, y_train, config, X_val, y_val, opts):
    """
    Main script for training an LSTM model.
    :param X_train: list of input training data (list of strings)
    :param y_train: list (or np array) of corresponding labels (list of integers)
    :param config: dict with the model configuration
    :param X_val: list of validation data (list of strings)
    :param y_val: list (or np array) of corresponding labels (list of integers)
    :param opts: Namespace object that contains the device
    :return:
    """
    print("Building training dataset...")
    # Build dataset
    train_data = ClfDataset(X=X_train,
                            y=y_train,
                            vocab_size=20000,
                            seq_len=config["data"]["seq_len"])
    train_data.truncate(10000)
    print("Building validation dataset...")
    val_data = ClfDataset(X=X_val,
                          y=y_val,
                          vocab=train_data.vocab,
                          seq_len=config["data"]["seq_len"])

    # Define a dataloader
    train_lengths = [len(x) for x in train_data.data]

    train_sampler = BucketBatchSampler(train_lengths, config["batch_size"], shuffle=True)
    train_loader = DataLoader(train_data, batch_sampler=train_sampler,
                              num_workers=config["num_workers"],
                              collate_fn=ClfCollate())
    val_loader = DataLoader(val_data, batch_size=config["batch_size"],
                            num_workers=config["num_workers"], shuffle=False,
                            collate_fn=ClfCollate())
    # the following is for calculating train loss and accuracy (forward pass/evaluate with train data)
    val_loader_train_dataset = DataLoader(train_data, shuffle=False,
                                          batch_size=config["batch_size"],
                                          num_workers=opts.cores,
                                          collate_fn=ClfCollate())
    # Define the model
    n_tokens = len(train_data.vocab)
    model = RNNClassifier(n_tokens,
                       nclasses=len(set(train_data.labels)),
                       **config["model"])
    criterion = nn.CrossEntropyLoss()

    # Load Pretrained Word Embeddings
    if "embeddings" in config["vocab"] and config["vocab"]["embeddings"]:
        emb_file = os.path.join(EMB_DIR, config["vocab"]["embeddings"])
        dims = config["vocab"]["embeddings_dim"]

        embs, emb_mask, missing = train_data.vocab.read_embeddings(emb_file, dims)
        model.initialize_embeddings(embs, config["model"]["embed_trainable"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(parameters,
                                 lr=config["lr"],
                                 weight_decay=config["weight_decay"])

    model.to(opts.device)
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters()
                                 if p.requires_grad)

    print("Total Params:", number_h(total_params))
    print("Total Trainable Params:", number_h(total_trainable_params))
    trainable_params = sorted([[n] for n, p in model.named_parameters()
                               if p.requires_grad])

    # Trainer: responsible for managing the training process
    trainer = ClfTrainer(model=model, train_loader=train_loader,
                         valid_loader=val_loader,
                         valid_loader_train_set=val_loader_train_dataset,
                         test_loader=None,
                         criterion=criterion,
                         optimizers=optimizer,
                         config=config, device=opts.device,
                         vocab=train_data.vocab,
                         batch_end_callbacks=None)

    # exp = Experiment(config["name"], config, output_dir=os.path.join(EXP_DIR, config["name"]))
    #
    # exp.add_metric("ep_loss", "line", "epoch loss", ["TRAIN", "VAL"])
    # exp.add_metric("ep_f1", "line", "epoch f1", ["TRAIN", "VAL"])
    # exp.add_metric("ep_acc", "line", "epoch accuracy", ["TRAIN", "VAL"])
    # exp.add_metric("ep_pre", "line", "epoch precision", ["TRAIN", "VAL"])
    # exp.add_metric("ep_rec", "line", "epoch recall", ["TRAIN", "VAL"])
    #
    # exp.add_value("epoch", title="epoch summary", vis_type="text")
    # exp.add_value("progress", title="training progress", vis_type="text")

    # training loop
    best_loss = None
    early_stopping = EarlyStopping("min", config["patience"])

    for epoch in range(config["epochs"]):
        train_loss = trainer.train_epoch()
        val_loss, y, y_pred = trainer.eval_epoch(val_set=True)
        _, y_train, y_pred_train = trainer.eval_epoch(train_set=True)

        # # Calculate accuracy and f1-macro on the evaluation set
        # cluster = True
        # if cluster is False:
        #     exp.update_metric("ep_loss", train_loss.item(), "TRAIN")
        #     exp.update_metric("ep_loss", val_loss.item(), "VAL")
        #
        #     exp.update_metric("ep_f1", f1_macro(y_train, y_pred_train), "TRAIN")
        #     exp.update_metric("ep_f1", f1_macro(y, y_pred), "VAL")
        #
        #     exp.update_metric("ep_acc", acc(y_train, y_pred_train), "TRAIN")
        #     exp.update_metric("ep_acc", acc(y, y_pred), "VAL")
        #
        #     exp.update_metric("ep_pre", precision_macro(y_train, y_pred_train), "TRAIN")
        #     exp.update_metric("ep_pre", precision_macro(y, y_pred), "VAL")
        #
        #     exp.update_metric("ep_rec", recall_macro(y_train, y_pred_train), "TRAIN")
        #     exp.update_metric("ep_rec", recall_macro(y, y_pred), "VAL")
        #
        #     print()
        #     epoch_log = exp.log_metrics(["ep_loss", "ep_f1", "ep_acc", "ep_pre", "ep_rec"])
        #     print(epoch_log)
        #     exp.update_value("epoch", epoch_log)
        #
        #     exp.save()
        # else:
        print()
        print("epoch: {}, train loss: {}, val loss: {}, accuracy: {}, precision: {},"
              "recall: {}, f1: {}".format(epoch, train_loss.item(), val_loss.item(),
                                          acc(y, y_pred),
                                          precision_macro(y, y_pred),
                                          recall_macro(y, y_pred),
                                          f1_macro(y, y_pred)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_loss or val_loss < best_loss:
            best_loss = val_loss
            trainer.best_val_loss = best_loss
            trainer.acc = acc(y, y_pred)
            trainer.f1 = f1_macro(y, y_pred)
            trainer.precision = precision_macro(y, y_pred)
            trainer.recall = recall_macro(y, y_pred)

            trainer.checkpoint(name=config["name"], verbose=False)

        if early_stopping.stop(val_loss):
            print("Early Stopping...")
            break

        print("\n")

    return model, train_data.vocab, criterion,  train_loss.item(),  val_loss.item(), \
           [f1_macro(y, y_pred), acc(y, y_pred), precision_macro(y, y_pred), recall_macro(y, y_pred)]

def test_lstm_model(X_test, y_test, vocab, config, opts, model, criterion):
    print("Building test dataset...")
    test_data = ClfDataset(X=X_test,
                           y=y_test,
                           vocab=vocab,
                           seq_len=config["data"]["seq_len"])
    # test_data.truncate(100)
    test_loader = DataLoader(test_data, batch_size=config["batch_size"],
                             num_workers=config["num_workers"], shuffle=False,
                             collate_fn=ClfCollate())

    trainer = ClfTrainer(model=model,
                         test_loader=test_loader,
                         config=config, device=opts.device,
                         criterion=criterion,
                         vocab=vocab,
                         batch_end_callbacks=None)
    test_loss, y, y_pred, prob_dist = trainer.eval_epoch(test_set=True)

    return test_loss, y, y_pred, prob_dist


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=False,
                        default='lstm_template',
                        help="config file of experiment (yaml)")
    parser.add_argument("-g", "--gpu", required=False,
                        default='0', help="gpu on which this experiment runs")

    args = parser.parse_args()
    input_config = args.input

    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # print("\nThis experiment runs on gpu {}...\n".format(args.gpu))

    opts, config = train_options(input_config, parser)

    # loop(opts, config)
    dataset_name = config["data"]["dataset"]
    X_train, y_train, X_val, y_val = load_dataset(dataset_name)

    model, vocab, criterion, train_loss, val_loss, scores = train_lstm_model(X_val, y_val,
                                                                      config, X_val, y_val, opts)
if __name__ == '__main__':
    main()
