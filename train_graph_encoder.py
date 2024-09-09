import argparse
import torch
import wandb
import os
import random
import json
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from accelerate import Accelerator
from time import strftime, localtime
from torch.utils.data import DataLoader
from torch_geometric.data import Batch
from torch_scatter import scatter_mean
from concurrent.futures import ThreadPoolExecutor, as_completed
from torchmetrics.classification import Accuracy
from torchmetrics.classification.auroc import AUROC
from src.models.gvp.modeling_gvp import AutoGraphEncoder
from src.utils.data_utils import BatchSampler, convert_graph
from src.utils.cluster_utils import norm_embed

NODE_S_TYPE_TO_NUM_LABELS = {
    'ss3': 3, 'ss8': 8, 'foldseek': 20, 'foldseek_ss8': 160, 
    'aa': 26, 'aa_foldseek': 520, 'aa_ss8': 208, 'aa_foldseek_ss8': 4160
}

def train(args, model, accelerator, train_loader, val_loader, optimizer=None, metrics=None, device=None):
    best_acc = 0
    val_acc_history = []
    path = os.path.join(args.ckpt_dir, args.model_name)
    for epoch in range(0):
        print(f"---------- Epoch {epoch} ----------")
        model.train()
        loss, acc = loop(args, model, accelerator, train_loader, epoch, optimizer, metrics, device)
        print(f'EPOCH {epoch} TRAIN loss: {loss:.4f} acc: {acc:.4f}')
        
        model.eval()
        with torch.no_grad():
            loss, acc = loop(args, model, accelerator, val_loader, epoch, metrics=metrics, device=device)
            if args.wandb:
                wandb.log({"valid/val_loss": loss, "valid/val_acc": acc, "valid/epoch": epoch})
        print(f'EPOCH {epoch} VAL loss: {loss:.4f} acc: {acc:.4f}')
        val_acc_history.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), path)
            print(f'>>> BEST at epcoh {epoch}, valid acc: {acc:.4f}')
            print(f'>>> Save model to {path}')
        
        if val_acc_history.index(max(val_acc_history)) < epoch - args.patience:
            print(f"Early stopping at epoch {epoch}")
            break
    
    
def loop(args, model, accelerator, dataloader, epoch, optimizer=None, metrics=None, device=None):
    total_loss, total_acc = 0, 0
    iter_num = len(dataloader)
    global_steps = epoch * len(dataloader)
    epoch_iterator = tqdm(dataloader)
    for batch in epoch_iterator:
        batch.to(device)
        h_V = (batch.node_s, batch.node_v)
        h_E = (batch.edge_s, batch.edge_v)
        
        loss, logits = model(h_V, batch.edge_index, h_E, batch.node_s_labels)
        total_loss += loss.item()
        
        acc = float(metrics(logits.squeeze(-1).cpu(), batch.node_s_labels.cpu()))
        total_acc += acc
        global_steps += 1
        
        if optimizer:
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            epoch_iterator.set_postfix(train_loss=loss.item(), train_acc=acc)
            if args.wandb:
                wandb.log({"train/train_loss": loss.item(), "train/train_acc": acc, "train/epoch": epoch}, step=global_steps)
        else:
            epoch_iterator.set_postfix(eval_loss=loss.item(), eval_acc=acc)
    
    epoch_loss = total_loss / iter_num
    epoch_acc = total_acc / iter_num
    return epoch_loss, epoch_acc


def get_emebeds(model, dataloader, device):
    epoch_iterator = tqdm(dataloader)
    embeds, names = [], []
    embed_dict = {}
    with torch.no_grad():
        for batch in epoch_iterator:
            batch.to(device)
            h_V = (batch.node_s, batch.node_v)
            h_E = (batch.edge_s, batch.edge_v)
            
            node_emebddings = model.get_embedding(h_V, batch.edge_index, h_E)
            graph_emebddings = scatter_mean(node_emebddings, batch.batch, dim=0).detach().cpu()
            embeds.append(graph_emebddings)
            names.extend(batch.name)
    
    embeds = torch.cat(embeds, dim=0)
    embeds = norm_embed(embeds)
    for name, embed in zip(names, embeds):
        embed_dict[name] = embed
    return embed_dict


if __name__== "__main__":
    parser = argparse.ArgumentParser()
    # training
    parser.add_argument('--ckpt_root', default='ckpt', help='root directory to save trained models')
    parser.add_argument('--ckpt_dir', default=None, help='directory to save trained models')
    parser.add_argument('--model_name', default=None, help='model name')
    parser.add_argument('--num_workers', type=int, default=4, help='number of threads for loading data')
    parser.add_argument('--max_batch_nodes', type=int, default=3000, help='max number of nodes per batch')
    parser.add_argument('--max_train_epochs', type=int, default=100, help='training epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--patience', type=int, default=10, help='patience for early stopping')
    
    # model
    parser.add_argument('--node_s_type', required=True, help='node scalar feature type')
    parser.add_argument('--num_layers', type=int, default=3, help='number of GVP layers')
    parser.add_argument('--num_labels', type=int, default=None, help='number of labels')
    parser.add_argument('--node_s_hidden_dim', type=int, default=256, help='node scalar feature hidden dim')
    parser.add_argument('--node_v_hidden_dim', type=int, default=32, help='node vector feature hidden dim')
    parser.add_argument('--edge_s_hidden_dim', type=int, default=64, help='edge scalar feature hidden dim')
    parser.add_argument('--edge_v_hidden_dim', type=int, default=2, help='edge vector feature hidden dim')
    
    # dataset
    parser.add_argument('--train_mask_node_ratio', type=float, default=1, help='ratio of masked nodes')
    parser.add_argument('--train_permute_node_ratio', type=float, default=0, help='ratio of permuted nodes')
    parser.add_argument('--train_graph_dir', required=True, help='directory of train graphs')
    parser.add_argument('--valid_graph_num', type=int, default=200, help='number of validation graphs')
    parser.add_argument('--test_graph_dir', help='directory of test graphs')
    parser.add_argument('--test_graph_pair_json', help='json file of test graph pairs')
    parser.add_argument('--test_mask_node', action='store_true', help='whether to mask node feature')
    
    # wandb log
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='subgraph')
    parser.add_argument("--wandb_entity", type=str, default="ty_ang")
    parser.add_argument('--wandb_run_name', type=str, default=None)
    args = parser.parse_args()
    
    assert args.train_permute_node_ratio + args.train_mask_node_ratio <= 1, "sum of train_permute_node_ratio and train_mask_node_ratio should be less than 1"
    
    os.makedirs(args.ckpt_root, exist_ok=True)
    if args.wandb:
        if args.wandb_run_name is None:
            args.wandb_run_name = f"graph_encoder_l{args.num_layers}_lr{args.lr}"
        wandb.init(
            project=args.wandb_project, name=args.wandb_run_name, 
            entity=args.wandb_entity, config=vars(args)
        )
    if args.model_name is None:
        args.model_name = f"{args.wandb_run_name}.pt"
    if args.ckpt_dir is None:
        current_date = strftime("%Y%m%d", localtime())
        args.ckpt_dir = os.path.join(args.ckpt_root, current_date)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    with open(os.path.join(args.ckpt_dir, args.model_name[:-3] + ".json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("---------- Build Model ----------")
    node_dim = (args.node_s_hidden_dim, args.node_v_hidden_dim)
    edge_dim = (args.edge_s_hidden_dim, args.edge_v_hidden_dim)
    if args.num_labels is None:
        args.num_labels = NODE_S_TYPE_TO_NUM_LABELS[args.node_s_type]
    model = AutoGraphEncoder(
        node_in_dim=(args.num_labels, 3), node_h_dim=node_dim, 
        edge_in_dim=(32, 1), edge_h_dim=edge_dim,
        num_layers=args.num_layers
    )
    model.to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"MODEL: {params:.2f}M parameters")
    
    print("---------- Prepare Dataset ----------")
    # train and valid dataset
    dataset, node_counts = [], []
    data_names = os.listdir(args.train_graph_dir)
    
    def process_subgraph(d):
        try:
            graph = torch.load(os.path.join(args.train_graph_dir, d))
        except:
            print(f"Error loading {d}")
            return None, None
        graph = convert_graph(graph)
        graph.name = d.split(".")[0]
        node_num = graph.node_s.shape[0]
        return graph, node_num
    
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(process_subgraph, d) for d in data_names]
        for future in tqdm(as_completed(futures), total=len(data_names)):
            graph, node_num = future.result()
            if graph is not None:
                dataset.append(graph)
                node_counts.append(node_num)

    random.shuffle(dataset)
    train_dataset, valid_dataset = dataset[:-args.valid_graph_num], dataset[-args.valid_graph_num:]
    trainset_node_counts, valset_node_counts = node_counts[:-args.valid_graph_num], node_counts[-args.valid_graph_num:]
    print(">>> trainset: ", len(train_dataset))
    print(">>> valset: ", len(valid_dataset))
    
    
    def collate_fn(data_list):
        batch = Batch.from_data_list(data_list)
        node_s_labels = torch.clone(batch.node_s).argmax(dim=-1)
        batch.node_s_labels = node_s_labels
        
        node_num = batch.node_s.shape[0]
        num_nodes_to_permute = int(args.train_permute_node_ratio * node_num)
        num_nodes_to_zero = int(args.train_mask_node_ratio * node_num)
        
        # select nodes to zero from all nodes
        nodes_to_zero = torch.randperm(node_num)[:num_nodes_to_zero]
        batch.node_s[nodes_to_zero] = 0
        
        # select nodes to permute from remaining nodes
        mask = torch.ones(node_num, dtype=torch.bool)
        mask[nodes_to_zero] = False
        remaining_nodes = torch.arange(node_num)[mask]
        nodes_to_permute = remaining_nodes[torch.randperm(remaining_nodes.size(0))[:num_nodes_to_permute]]
        
        # generate one hot vectors
        one_hot_vectors = torch.zeros((num_nodes_to_permute, args.num_labels))
        indices = torch.randint(0, args.num_labels, (num_nodes_to_permute,))
        one_hot_vectors[torch.arange(num_nodes_to_permute), indices] = 1
        
        # permute node labels
        batch.node_s[nodes_to_permute] = one_hot_vectors
        
        return batch
    
    
    train_loader = DataLoader(
        train_dataset, num_workers=args.num_workers,
        batch_sampler=BatchSampler(
                        trainset_node_counts, 
                        max_batch_nodes=args.max_batch_nodes
                        ),
        collate_fn=collate_fn
        )
    valid_loader = DataLoader(
        valid_dataset, num_workers=args.num_workers,
        batch_sampler=BatchSampler(
                        valset_node_counts, 
                        max_batch_nodes=args.max_batch_nodes,
                        shuffle=False
                        ),
        collate_fn=collate_fn
        )
    
    metrics = Accuracy(task="multiclass", num_classes=args.num_labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    accelerator = Accelerator()
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, valid_loader
    )
    
    print("---------- Smple 3 data point from trainset ----------")
    for i in random.sample(range(len(train_dataset)), 3):
        print(">>> ", train_dataset[i])
    
    print("---------- Start Training ----------")
    train(args, model, accelerator, train_loader, val_loader,
          optimizer, metrics, device=device)
    
    
    # test dataset
    if args.test_graph_dir:
        model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, args.model_name)))
        test_dataset, test_node_counts = [], []
        test_pair = json.load(open(args.test_graph_pair_json, "r"))
        neg_names = ["negative", "extreme_neg", "hard_neg", "simple_neg", "easy_neg"]
        postive_samples = test_pair["positive"]
        
        print("---------- Prepare Test Dataset ----------")
        data_names = os.listdir(args.test_graph_dir)
        
        def process_graph(d):
            try:
                graph = torch.load(os.path.join(args.test_graph_dir, d))
            except Exception as e:
                print(e)
                print(f"Error loading {d}")
                return None, None
            graph = convert_graph(graph)
            graph.name = d.split(".")[0]
            node_num = graph.node_s.shape[0]
            return graph, node_num
        
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_graph, d) for d in data_names]
            for future in tqdm(as_completed(futures), total=len(data_names)):
                graph, node_num = future.result()
                test_dataset.append(graph)
                test_node_counts.append(node_num)
        
        def test_collate_fn(data_list):
            batch = Batch.from_data_list(data_list)
            if args.test_mask_node:
                batch.node_s = torch.zeros_like(batch.node_s)
            return batch
        
        test_loader = DataLoader(
            test_dataset, num_workers=args.num_workers,
            batch_sampler=BatchSampler(
                            test_node_counts, 
                            max_batch_nodes=args.max_batch_nodes,
                            shuffle=False
                            ),
            collate_fn=test_collate_fn
            )
        
        auroc_metrics = AUROC(task="binary")
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        print("---------- Start Testing ----------")
        embeds = get_emebeds(model, test_loader, device)
        for neg in neg_names:
            y_label = [1]* len(postive_samples) + [0]*len(test_pair[neg])
            y_true = []
            y_pred = []
            samples = postive_samples + test_pair[neg]
            for idx, sample in enumerate(samples):
                pdb1, pdb2, match_level = sample
                try:
                    sim_score = float(torch.mm(embeds[pdb1].unsqueeze(0), embeds[pdb2].unsqueeze(1)))
                except:
                    continue
                y_true.append(y_label[idx])
                y_pred.append(sim_score)
            auroc = float(auroc_metrics(torch.tensor(y_pred), torch.tensor(y_true)))
            print(f"Test AUROC for postive vs {neg}: {auroc}")
            if args.wandb:
                wandb.log({f"test/pos_{neg}_auroc": auroc})
    
    if args.wandb:
        wandb.finish()