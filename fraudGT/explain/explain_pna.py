import torch
import fraudGT  # noqa, register custom modules
from torch_geometric.explain import ModelConfig, Explainer
from torch_geometric.explain.explanation import _visualize_score
import os

from fraudGT.graphgym.config import cfg, set_cfg, load_cfg
from fraudGT.graphgym.cmd_args import parse_args
from fraudGT.graphgym.model_builder import create_model
from fraudGT.graphgym.checkpoint import load_ckpt
from fraudGT.explain.model_wrapper import PNAExplainerWrapper
from fraudGT.explain.edge_feat_gnn_explainer import EdgeFeatGNNExplainer
from fraudGT.loader.master_loader import load_dataset_master
from fraudGT.graphgym.loader import create_loader

EDGE_FEAT_LABELS_ETH = ['Timestamp', 'Value', 'Nonce', 'Block Nr', 'Gas', 'Gas Price', 'Transaction Type', 'Port']

# Visualise edge features importance (adaption of existing visualise node feature importance func.)
def visualize_edge_feat_importance(edge_feat_mask, path=None, top_k=None):
    score = edge_feat_mask.sum(dim=0)
    _visualize_score(score, EDGE_FEAT_LABELS_ETH, path, top_k)

# Config loading
# Read command line args (specifically, path to pna config)
args = parse_args()
# Initialise default config then load yaml on top
set_cfg(cfg)
load_cfg(cfg, args)
cfg.freeze()

# Load dataset & model
dataset, _ = load_dataset_master(cfg.dataset.format, cfg.dataset.dir, cfg) # Returns dataset and splits
model = create_model(dataset=dataset)

# Load checkpoint (trained model weights)
load_ckpt(model, optimizer=None, scheduler=None)
model.eval() # Put in eval mode (displaces dropout etc)

# Get a single batch from the dataloader
loaders = create_loader(dataset) # Creates train, val and test loaders
test_loader = loaders[2]

# Take a single batch
batch = next(iter(test_loader))
batch = batch.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # Move batch to wherever model is. Batch comes form the datalaoder on CPU by default.

# Identify target node for explanations
task_node = cfg.dataset.task_entity # Reads "node" from config (renaming as task_node to avoid confusion)
pred = model(batch)[0] # Returns (pred, label) - > just take pred. Gives tensor of shape (N,1)


phishing_scores = pred.squeeze()
target_nodes = phishing_scores.topk(20).indices # Top 20 highest phishing prob nodes are used for explanations


# Build wrapper and explainer
wrapper = PNAExplainerWrapper(model, batch)
explainer = Explainer(
    model = wrapper,
    algorithm = EdgeFeatGNNExplainer(),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=ModelConfig(
        mode='binary_classification',
        task_level='node',
        return_type='raw', # before applying sigmoid
    )
)

# Run Explanations
os.makedirs(f'results/explanations/', exist_ok=True) # Create that directory if it doesn't already exist to stop crashing
for i, node_idx in enumerate(target_nodes):
    explanation = explainer(
        model = wrapper,
        x = batch.collect('x'), # Collect node features and store in plain dict for explainer
        edge_index=batch.collect('edge_index'),
        target=batch[task_node].y,
        index=node_idx,
        edge_attr=batch.collect('edge_attr')
    )
    torch.save(explanation, f'results/explanations/node_{node_idx}.pt') # Save explanation (node_mask, edge_mask & edge_feat_mask)
    if i < 5:
        explanation.visualize_graph(path=f'results/explanations/graph_node_{node_idx}.png')

# Aggregate edge feature importance across all 20 nodes for feature importance bar chart
all_masks = [torch.load(f'results/explanations/node_{node_idx}.pt').edge_feat_mask[('node', 'to', 'node')] for node_idx in target_nodes]
avg_mask = torch.stack(all_masks).mean(dim=0)
visualize_edge_feat_importance(avg_mask, path='results/explanations/edge_feat_aggregated.png')






