"""
Logic Gate Neural Networks are Good for Verification
Accepted at NeuS ’25
Paper: Kresse, F., Yu, E., Lampert, C. H., & Henzinger, T. A. (2025).

Code authors:
    Fabian Kresse (corresponding) – fabian.kresse@ist.ac.at
    Emily Yu – emily.yu@ist.ac.at

Description:
    Script to train the logic gate neural network on various datasets.
"""

import torch
import numpy as np
import json 
import os
from tqdm import tqdm
from datasets_neus import dataset
from torch import nn
from utils.exp_setup import build_model


def eval_model(model, loader, mode, device='cpu'):
    orig_mode = model.training
    with torch.no_grad():
        model.train(mode=mode)
        res = np.mean(
            [
                (model(x.to(device).round()).argmax(-1) == y.to(device)).to(torch.float32).mean().item()
                for x, y in loader
            ]
        )
        model.train(mode=orig_mode)
    return res.item()

def count_preds_vs_truth(model, loader, n_classes, device="cpu"):
    """
    Useful for manually checking if each class gets picked.
    """
    was_training = model.training           # remember original mode
    model.eval()                            # run in eval mode
    
    pred_cnt  = torch.zeros(n_classes, dtype=torch.long)
    true_cnt  = torch.zeros(n_classes, dtype=torch.long)

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(-1)

            pred_cnt += torch.bincount(preds.cpu(), minlength=n_classes)
            true_cnt += torch.bincount(y.cpu(),   minlength=n_classes)

    model.train(was_training)               # restore original mode

    print("is (predicted):", pred_cnt)
    print("should be    :", true_cnt)
    return pred_cnt, true_cnt 

def train(model, x, y, loss_fn, optimizer):
    outputs = model(x)
    loss = loss_fn(outputs, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()

def main(experiment="adult", 
         k=100, 
         seed=42, 
         path='models'):
    device = "cpu" #torch.device("cuda" if torch.cuda.is_available() else "cpu")
    k = k
    loaders, _,_,C, in_dim = dataset.build_dataset(experiment, 
                                                    batch_size=128, 
                                                    seed= seed)
    # print(loaders)
    train_loader = loaders.train
    test_loader = loaders.test
    val_loader = loaders.val
    
    model = build_model(in_dim, C=C, k=k, l=3, device=device)
    optim = torch.optim.Adam(model.parameters(), lr=0.01)

    # assign some weights to make sure we dont have degenerate solutions (some classes never picked)
    # !law with 50 gates per layer sometimes has degenerate solutions without weights!
    # if experiment=='law':
    #    weight = torch.tensor([1/0.1, 1/0.9])
    # elif experiment == 'folktable_5':
    #    weight = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0])
    # else:
    #    weight = torch.ones(C)
    
    weight = torch.ones(C).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    print(model)
    
    model.to(device)
    all_acuracies = []
    epochs = 201 # 201 epochs in paper
    progress_bar = tqdm(range(epochs), desc="Training Progress")

    for i in progress_bar:
        model.train()
        total_loss = 0
        for batch in train_loader:
            x_batch, y_batch = batch
            x = x_batch.to(device).float()
            y = y_batch.to(device)
            
            loss = train(
                model, x, y, loss_fn, optim
            )
            total_loss += loss 
            
            
        if i % 10 == 0:
            test_acc = eval_model(model, test_loader, mode=False, device=device)
            train_acc = eval_model(model, train_loader, mode=False, device=device)
            test_acc_prob = eval_model(model, train_loader, mode=True, device=device)
            val_acc = eval_model(model, val_loader, mode=False, device=device)
            count_preds_vs_truth(model, test_loader, C, device=device)
            print(f"Test accuracy: {test_acc}")
            print(f"Train accuracy: {train_acc}")
            print(f"Train accuracy (prob): {test_acc_prob}")
            print(f"Total loss: {total_loss}")

            all_acuracies.append((train_acc, test_acc, val_acc, test_acc_prob))
            tqdm.write(
                f"Epoch {i}: "
                f"Test Accuracy: {test_acc:.4f}, "
                f"Train Accuracy: {train_acc:.4f}, "
                f"Train Accuracy (Prob): {test_acc_prob:.4f}, "
                f"Total Loss: {total_loss:.4f}, "
            )
        progress_bar.set_postfix(
            {
                "Loss": total_loss / len(train_loader),
                "test_acc": test_acc,
                "train_acc": train_acc,
                "train_acc_prob": test_acc_prob,
                "val_acc": val_acc,
            }
        )
    
    # convert the model to a base_model
    train_accs, test_accs, val_accs, test_acc_probs = zip(*all_acuracies)
    accuracy_dict = {
        "train_acc": list(train_accs),
        "test_acc": list(test_accs),
        "val_acc": list(val_accs),
        "test_acc_prob": list(test_acc_probs),
    }
    os.makedirs(path, exist_ok=True)
    model_save_path = f"{path}/model_{experiment}_{k}_{seed}.pth"
    torch.save(model.state_dict(), model_save_path)


    file_name = f"{path}/acc_{experiment}_{k}_{seed}.json"
    
    with open(file_name, "w") as f:
        json.dump(accuracy_dict, f, indent=4)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["adult", "german_credit", "folktable_5", "law", "compas"],
        default="adult",
        help="Which dataset/experiment to run"
    )
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    # assert that if compas is selected, k is % 3 == 0
    assert args.experiment != "compas" or args.k % 3 == 0, "k must be divisible by 3 for compas dataset"
    main(experiment=args.experiment, 
         k=args.k, 
         seed=args.seed, 
         path = 'models')
