"""
This script is the main entry point for this repository. It allows you to:
1. load and preprocess a relational database into a graph format,
2. train a Graph-Conditional Relational Diffusion Model (GRDM) to learn the attribute distribution conditioned on the graph structure,
3. sample a synthetic graph structure using a node degree-preserving random graph generation algorithm,
4. sample synthetic node attributes conditioned on the sampled graph structure using the trained GRDM,
5. evaluate the quality of the generated synthetic relational database using multiple metrics.

For details about the method, please refer to the paper: https://arxiv.org/abs/2505.16527
"""

import argparse
import os
import sys
import time
from copy import deepcopy

import torch

import wandb
from rdbdiff.dataloader import get_neighbor_dataloader
from rdbdiff.dataset import RDBDiffDataset
from rdbdiff.model import GRDM, GNNDenoiser
from rdbdiff.random_graph_sampler import RandomGraphSampler
from rdbdiff.sampler import GRDMSampler, SamplingBudgetExhausted
from rdbdiff.third_party.clava_ddpm import gen_multi_report
from rdbdiff.trainer import GRDMTrainer, TrainingBudgetExhausted
from rdbdiff.utils import (
    count_params,
    decode_onehot_synthetic_data,
    denormalize_data,
    extract_attributes,
    get_col_to_stype_dict,
    make_pkey_fkey_graph,
    manual_seed,
    preprocess_database,
    read_json,
    read_pickle,
    round_synthetic_data,
    write_database,
    write_pickle,
)

#########################
### 0. Config Loading ###
#########################

parser = argparse.ArgumentParser()
parser.add_argument(
    "config_path", type=str, help="Path to the dataset-specific json config file."
)
# general args
parser.add_argument(
    "--exp_name",
    type=str,
    default="default_run",
    help="Descriptive experiment name used to separate logging and model checkpoints of different experiments on the same dataset",
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument(
    "--fixed_structure",
    action="store_true",
    help="Whether to use the training graph structure instead of sampling a new one (for ablation)",
)
parser.add_argument(
    "--no_wandb",
    action="store_true",
    help="Whether to disable wandb logging, useful when debugging.",
)
parser.add_argument(
    "--categorical_decoding",
    type=str,
    default="round",
    choices=["round", "match"],
    help=(
        "How to turn diffused categorical values back into class codes. 'round' is the "
        "upstream behaviour (round to nearest code). 'match' places the thresholds so the "
        "synthetic marginal matches the real one, which stops rare interior classes from "
        "being absorbed by their denser neighbours (fix S1). Non-default values write to a "
        "separate output directory, leaving the baseline run untouched."
    ),
)
# dataloader args
parser.add_argument(
    "--train_batch_size",
    type=int,
    default=4096,
    help="Training batch size. Can be reduced to reduce memory requirements.",
)
parser.add_argument(
    "--n_hops",
    type=int,
    default=1,
    help="Number of hops to sample in each subgraph. Will also be used to set GNN layers.",
)
parser.add_argument(
    "--num_neighbors",
    nargs="+",
    type=int,
    default=[-1],
    help="Number of neighbors to sample at each hop. Set -1 to sample all neighbors.",
)
parser.add_argument(
    "--cpu",
    action="store_true",
    help="Whether to use CPU instead of GPU (default). Useful for debugging.",
)
parser.add_argument(
    "--categorical_encoding",
    type=str,
    default="ordinal",
    choices=["ordinal", "onehot"],
    help=(
        "How categorical columns enter the model (fix S4). 'ordinal' is the upstream "
        "behaviour: raw class codes, quantile-transformed alongside the numerical columns and "
        "diffused as continuous values, which gives the model a false metric in which a rare "
        "class between two dense ones is absorbed. 'onehot' expands each categorical into one "
        "indicator column per class, skips the quantile transform for them, and decodes by "
        "argmax."
    ),
)
parser.add_argument(
    "--categorical_diffusion",
    type=str,
    default="gaussian",
    choices=["gaussian", "multinomial"],
    help=(
        "How categorical columns are diffused. 'gaussian' is the upstream behaviour (they "
        "ride along in the continuous vector). 'multinomial' lays them out as one-hot "
        "indicator columns (like --categorical_encoding onehot) but denoises them in log "
        "space with a true discrete diffusion process -- never a Gaussian over the simplex, "
        "which is what makes it different from S4. This is the principled route for the "
        "loan.status class-B collapse. Requires --categorical_encoding ordinal (the default)."
    ),
)
parser.add_argument(
    "--max_hours",
    type=float,
    default=None,
    help=(
        "Wall-clock budget for this process, in hours. Training stops at the next epoch "
        "boundary and sampling at the next checkpointed timestep once it is spent, then "
        "the process exits with code 42 having saved a resume point. Set it below this "
        "node's ~24h process lifetime so a run is never SIGKILLed mid-stage; relaunching "
        "resumes where it stopped. Omit for no limit."
    ),
)
parser.add_argument(
    "--gnn_aggr",
    nargs="+",
    default=None,
    help=(
        "Neighbourhood aggregation, overriding the config's model.gnn_aggr (fix S8). One "
        "name reproduces upstream. Several (e.g. `--gnn_aggr sum mean max std`) make each "
        "SAGEConv a MultiAggregation that concatenates the statistics: a downstream model "
        "builds its features from counts, means, extremes and spreads of a child table, and "
        "a denoiser conditioned on one moment of a neighbourhood cannot reproduce the rest."
    ),
)
parser.add_argument(
    "--gnn_residual",
    action="store_true",
    help=(
        "Add residual connections around every message-passing layer and drop the ReLU on "
        "the last one (fix S9). Upstream rewrites the representation at each hop, which is "
        "why depth has been a trade rather than an addition -- S2 bought the cross-table "
        "task 18 points and cost the single-parent task 17."
    ),
)
parser.add_argument(
    "--degree_features",
    action="store_true",
    help=(
        "Condition each row on its exact per-relation child counts (fix S10). Neighbour "
        "sampling caps the fan-out the GNN sees, so its aggregates are fixed-size samples "
        "that carry no cardinality information; the structure is sampled before any "
        "attribute is generated, so the true counts are available at generation time."
    ),
)
parser.add_argument(
    "--self_conditioning",
    action="store_true",
    help=(
        "Run the denoiser in two passes, the second conditioned on the first's x0 estimate "
        "of every node (fix S11). While a row is generated its neighbours sit at the same "
        "noise level, so the measured conditioning window is one hop behind the nominal "
        "depth (docs/downstream/DIAGNOSIS.md A2); showing the second pass denoised neighbours buys that hop "
        "back at roughly the cost of one extra GNN forward rather than one extra hop."
    ),
)
parser.add_argument(
    "--self_cond_aux_weight",
    type=float,
    default=1.0,
    help=(
        "Weight of the x0-reconstruction loss that supervises the first pass. The estimate "
        "is fed forward detached, so this term is the only thing that trains that head."
    ),
)
parser.add_argument(
    "--seed_balance_alpha",
    type=float,
    default=0.0,
    help=(
        "Re-weight seed-node sampling by count(node_type) ** -alpha (fix S3). 0.0 is the "
        "upstream behaviour (uniform over rows, so on Berka `trans` takes 4008 of every 4096 "
        "seeds and `loan` takes 2.6); 1.0 samples node types equally. Intermediate values "
        "temper the correction."
    ),
)
args = parser.parse_args()

# set random seed for reproducibility
seed = args.seed
manual_seed(seed)

# extract args
# general args
config_path = args.config_path
exp_name = args.exp_name
generate_structure = not args.fixed_structure
# dataloader args
train_batch_size = args.train_batch_size
n_hops = args.n_hops
num_neighbors = args.num_neighbors
assert len(num_neighbors) == n_hops, "num_neighbors length must match n_hops"
use_cpu = args.cpu
# The one-hot column layout is shared by fix S4 and the multinomial diffusion path; only the
# diffusion process over that layout differs. `onehot_layout` gates every place the raw class
# codes would otherwise be quantile-transformed / rounded.
assert not (
    args.categorical_diffusion == "multinomial" and args.categorical_encoding == "onehot"
), "--categorical_diffusion multinomial expects --categorical_encoding ordinal"
onehot_layout = (
    args.categorical_encoding == "onehot" or args.categorical_diffusion == "multinomial"
)

# load config
config = read_json(config_path)
dataset_name = config["data"]["dataset_name"]
data_path = config["data"]["data_path"]
dimension_tables = config["data"]["dimension_tables"]
if len(dimension_tables) == 0:
    dimension_tables = None
# model args
gnn_aggr = args.gnn_aggr if args.gnn_aggr else config["model"]["gnn_aggr"]
if isinstance(gnn_aggr, list) and len(gnn_aggr) == 1:
    gnn_aggr = gnn_aggr[0]  # a single name is the upstream single-aggregator path
d_layers = config["model"]["d_layers"]
dropout = config["model"]["dropout"]
dim_t = config["model"]["dim_t"]
# diffusion args
num_timesteps = config["diffusion"]["num_timesteps"]
gaussian_loss_type = config["diffusion"]["gaussian_loss_type"]
scheduler = config["diffusion"]["scheduler"]
# trainer args
lr = config["trainer"]["lr"]
weight_decay = config["trainer"]["weight_decay"]
steps = config["trainer"]["steps"]
# sampler args
sample_batch_size = config["sampler"]["sample_batch_size"]
size_multiplier = config["sampler"]["size_multiplier"]

# prepare workspace directory and device
deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
workspace_dir = f"rdbdiff_workspace/{dataset_name}/{exp_name}/seed_{seed}"
os.makedirs(workspace_dir, exist_ok=True)
device = (
    torch.device("cuda")
    if torch.cuda.is_available() and not use_cpu
    else torch.device("cpu")
)

# initialize wandb
print("Initializing W&B...")
wandb.init(
    project="RDBDiff",
    group=dataset_name,
    name=f"{dataset_name}_{exp_name}_seed_{seed}",
    mode="disabled" if args.no_wandb else None,
)

# print all args
print("General args:")
print(f"  config_path: {config_path}")
print(f"  exp_name: {exp_name}")
print(f"  seed: {seed}")
print(f"  generate_structure: {generate_structure}")
print("Dataloader args:")
print(f"  train_batch_size: {train_batch_size}")
print(f"  n_hops: {n_hops}")
print(f"  num_neighbors: {num_neighbors}")
print("Data args:")
print(f"  dataset_name: {dataset_name}")
print(f"  data_path: {data_path}")
print(f"  dimension_tables: {dimension_tables}")
print("Model args:")
print(f"  gnn_aggr: {gnn_aggr}")
print(f"  gnn_residual: {args.gnn_residual}")
print(f"  degree_features: {args.degree_features}")
print(f"  self_conditioning: {args.self_conditioning}")
if args.self_conditioning:
    print(f"  self_cond_aux_weight: {args.self_cond_aux_weight}")
print(f"  d_layers: {d_layers}")
print(f"  dropout: {dropout}")
print(f"  dim_t: {dim_t}")
print("Diffusion args:")
print(f"  num_timesteps: {num_timesteps}")
print(f"  gaussian_loss_type: {gaussian_loss_type}")
print(f"  scheduler: {scheduler}")
print("Trainer args:")
print(f"  lr: {lr}")
print(f"  weight_decay: {weight_decay}")
print(f"  steps: {steps}")
print("Sampler args:")
print(f"  sample_batch_size: {sample_batch_size}")
print(f"  size_multiplier: {size_multiplier}")
print("Other args:")
print(f"  workspace_dir: {workspace_dir}")
print(f"  device: {device}")

###########################
### 1. Data Preparation ###
###########################

# load data from disk
print("Loading data...")
dataset_real = RDBDiffDataset(
    data_path, cache_dir=None
)  # use cache_dir if needed for large data
database_real = dataset_real.get_db()
dataset_meta = dataset_real.dataset_meta

# preprocess data: normalize values using quantile transform
print("Preprocessing data...")
# TODO(AK): assert we can reconstruct all/most data
normalizers_dir = os.path.join(workspace_dir, "normalizers")
preprocess_database(
    database_real,
    normalizers_dir,
    seed,
    data_path=data_path,
    categorical_encoding=args.categorical_encoding,
    onehot_layout=onehot_layout,
)

# construct graph
print("Constructing graph...")
col_to_stype_dict = get_col_to_stype_dict(database_real, data_path)
# Class counts per categorical column: sizes the one-hot blocks for fix S4 and for the
# multinomial diffusion path.
col_to_num_classes = (
    {
        table_name: {
            col: info["size"]
            for col, info in read_json(
                os.path.join(data_path, f"{table_name}_domain.json")
            ).items()
            if info["type"] == "discrete"
        }
        for table_name in database_real.table_dict
    }
    if onehot_layout
    else None
)
graph_real = make_pkey_fkey_graph(
    database_real,
    col_to_stype_dict=col_to_stype_dict,  # speficied column types
    col_to_num_classes=col_to_num_classes,
    add_degree_features=args.degree_features,
)
# NOTE 1: make_pkey_fkey_graph removes primary and foreign key columns from col_to_stype_dict
# NOTE 2: from this point onwards, we view the data as a graph so we use graph terminology
# e.g. node_type refers to table_name etc.

# get dataloader
node_type_to_d_in = {
    node_type: graph_real[node_type].x.shape[1] for node_type in graph_real.node_types
}

# For multinomial categorical diffusion the denoiser output splits at n_num into an eps head
# and a per-class-logit tail. The per-column class counts must be listed in the same order the
# one-hot blocks were concatenated in make_pkey_fkey_graph (categorical columns of
# col_to_stype_dict, whose pkey/fkey entries it has just removed).
node_type_to_n_num = {
    node_type: int(graph_real[node_type].num_numerical_features[0].item())
    for node_type in graph_real.node_types
}
# Per-node-type ordered class counts, aligned with the one-hot block layout. Built whenever
# the one-hot layout is active so the decode below uses the same order the blocks were
# concatenated in; passed to GRDM only for the multinomial diffusion path.
node_type_to_cat_num_classes = None
if onehot_layout:
    node_type_to_cat_num_classes = {
        node_type: [
            col_to_num_classes[node_type][col]
            for col, stype in col_to_stype_dict[node_type].items()
            if stype == "categorical"
        ]
        for node_type in graph_real.node_types
        if node_type in col_to_num_classes
    }
node_type_to_num_classes = (
    node_type_to_cat_num_classes
    if args.categorical_diffusion == "multinomial"
    else None
)
if node_type_to_num_classes is not None:
    print(f"multinomial num_classes per node type: {node_type_to_num_classes}")
train_dataloader, n_total_input_nodes = get_neighbor_dataloader(
    data=graph_real,
    batch_size=train_batch_size,
    shuffle=True,
    disjoint=True,
    num_neighbors=num_neighbors,
    dimension_tables=dimension_tables,
    node_type_to_d_in=node_type_to_d_in,
    seed_balance_alpha=args.seed_balance_alpha,
)

########################
### 2. GRDM Training ###
########################

# prepare model
gnn_params = {
    "node_types": graph_real.node_types,
    "edge_types": graph_real.edge_types,
    "aggr": gnn_aggr,
    "num_layers": n_hops,  # consistent with the number of hops in the dataloaders
}
rtdl_params = {
    "dropout": dropout,
    "d_layers": d_layers,
}
# set the number of layers in the RTDL MLPs based on the size of each table
node_type_to_rtdl_d_layers = {}
for node_type in graph_real.node_types:
    if dimension_tables is not None and node_type in dimension_tables:
        continue
    if graph_real[node_type].num_nodes >= 10000:
        node_type_to_rtdl_d_layers[node_type] = d_layers
    else:
        node_type_to_rtdl_d_layers[node_type] = [512, 1024, 512]
print(f"node_type_to_rtdl_d_layers: {node_type_to_rtdl_d_layers}")

print(f"Instantiating diffusion model...")
# denoising model
# the degree block has one column per edge type and the same width on every node type, so
# any node type reports the right size
degree_dim = (
    graph_real[graph_real.node_types[0]].deg.shape[1] if args.degree_features else 0
)
denoising_model = GNNDenoiser(
    node_type_to_d_in,
    gnn_params,
    rtdl_params,
    dim_t,
    dimension_tables=dimension_tables,
    node_type_to_rtdl_d_layers=node_type_to_rtdl_d_layers,
    gnn_residual=args.gnn_residual,
    degree_dim=degree_dim,
    self_conditioning=args.self_conditioning,
)
denoising_model.to(device)

# diffusion model
grdm = GRDM(
    denoise_fn=denoising_model,
    num_timesteps=num_timesteps,
    gaussian_loss_type=gaussian_loss_type,
    scheduler=scheduler,
    device=device,
    dimension_tables=dimension_tables,
    self_cond_aux_weight=args.self_cond_aux_weight,
    categorical_diffusion=args.categorical_diffusion,
    node_type_to_num_classes=node_type_to_num_classes,
    node_type_to_n_num=node_type_to_n_num,
)
grdm.to(device)
grdm.train()

print(f"TOTAL Number of parameters in GRDM: {count_params(grdm)}")
print(f"Number of parameters in denoising model: {count_params(denoising_model)}")
print(f"Number of parameters of GNN: {count_params(denoising_model.gnn)}")
print(f"Number of parameters of MLPs: {count_params(denoising_model.mlps)}")
print(
    f"Number of parameters of time embedding: {count_params(denoising_model.time_embed)}"
)
print(
    f"Number of parameters of input projections: {count_params(denoising_model.projs)}"
)
wandb.log({"model/num_params": count_params(grdm)["params-total"]})

# training loop
model_ckpt_dir = os.path.join(workspace_dir, "model_checkpoint")
trainer = GRDMTrainer(
    diffusion_model=grdm,
    train_dataloader=train_dataloader,
    lr=lr,
    weight_decay=weight_decay,
    steps=steps,
    save_dir=model_ckpt_dir,
    device=device,
    train_batch_size=train_batch_size,
    n_total_target_nodes=n_total_input_nodes,
    max_epochs=None,
    target_node_types=[
        nt
        for nt in graph_real.node_types
        if dimension_tables is None or nt not in dimension_tables
    ],
    seed_balance_alpha=args.seed_balance_alpha,
    deadline=deadline,
)

if not trainer.is_model_already_trained():
    if trainer.has_checkpoint():
        start_step = trainer.load_checkpoint()
        print(f"Resuming training from checkpoint at step {start_step}...")
    else:
        start_step = 0
        print(f"Training model...")
    try:
        trainer.run_training_loop(start_step=start_step)
    except TrainingBudgetExhausted:
        # checkpoint is written; code 42 tells the relauncher this is a planned stop,
        # not a crash, so it should start the process again rather than give up
        sys.exit(42)
    trainer.save_model_and_loss_history()
else:
    print(f"Loading trained model...")
    trainer.load_model()

# use the EMA model for sampling
grdm._denoise_fn = trainer.ema_model
grdm.eval()

# once training is done, we can sample from the diffusion models
sampler = GRDMSampler(
    diffusion_model=grdm,
    save_dir=model_ckpt_dir,
    num_timesteps=num_timesteps,
    num_neighbors=num_neighbors,
    sample_batch_size=sample_batch_size,
    device=device,
    dimension_tables=dimension_tables,
    node_type_to_d_in=node_type_to_d_in,
    deadline=deadline,
)

#####################################
### 3. Graph Structure Generation ###
#####################################

# sampling
if generate_structure:
    syn_data_dir = os.path.join(
        workspace_dir,
        f"synthetic_data_sampled_structure_size_multiplier_{size_multiplier}"
        + (
            ""
            if args.categorical_decoding == "round"
            else f"_decoding_{args.categorical_decoding}"
        ),
    )
    os.makedirs(syn_data_dir, exist_ok=True)

    sampled_structure_dir = os.path.join(syn_data_dir, "sampled_structure")
    if not os.path.exists(sampled_structure_dir):
        print("Sampling structure...")
        structure_sampler = RandomGraphSampler(
            real_data_path=data_path,
            dimension_tables=dimension_tables,
            keep_pkey_fkeys=None,
        )
        structure_sampler.sample_and_save_structure(
            seed=seed,
            size_multiplier=size_multiplier,
            save_dir=sampled_structure_dir,
        )
    else:
        print("Sampled structure is already generated. Loading it...")

    # load in both cases
    database_syn = RDBDiffDataset(sampled_structure_dir, cache_dir=None).get_db()
    preprocess_database(
        database_syn,
        normalizers_dir,
        seed,
        data_path=data_path,
        categorical_encoding=args.categorical_encoding,
        onehot_layout=onehot_layout,
    )  # VERY IMPORTANT when using dimension tables
    graph_empty_attributes = make_pkey_fkey_graph(
        database_syn,
        col_to_stype_dict=col_to_stype_dict,
        col_to_num_classes=col_to_num_classes,
        add_degree_features=args.degree_features,
    )
else:  # fixed structure
    syn_data_dir = os.path.join(workspace_dir, "synthetic_data_real_structure")
    os.makedirs(syn_data_dir, exist_ok=True)

    graph_empty_attributes = deepcopy(graph_real)
    for node_type in graph_empty_attributes.node_types:
        if dimension_tables is not None and node_type in dimension_tables:
            continue
        graph_empty_attributes[node_type].x = torch.zeros_like(
            graph_empty_attributes[node_type].x
        )

################################################
### 4. Attribute Sampling and Postprocessing ###
################################################

X_syn_path = os.path.join(syn_data_dir, "node_type_to_X_syn.pkl")
if not os.path.exists(X_syn_path):
    print("Sampling...")
    try:
        graph_syn = sampler.sample_node_attributes(graph_empty_attributes)
    except SamplingBudgetExhausted:
        sys.exit(42)

    node_type_to_X_syn = {
        node_type: graph_syn[node_type].x.detach().cpu().numpy()
        for node_type in graph_syn.node_types
    }
    write_pickle(node_type_to_X_syn, X_syn_path)
else:
    print("Synthetic data is already sampled. Loading...")
    node_type_to_X_syn = read_pickle(X_syn_path)

# number of numerical features per table (the categorical block is whatever follows)
node_type_to_num_numerical_features = {
    node_type: graph_real[node_type].num_numerical_features[0].item()
    for node_type in graph_real.node_types
}

# denormalize synthetic data
node_type_to_X_syn = denormalize_data(
    node_type_to_X_syn,
    normalizers_dir,
    node_type_to_num_numerical_features=(
        node_type_to_num_numerical_features if onehot_layout else None
    ),
)

# re-load raw data (without preprocessing)
database_raw = RDBDiffDataset(data_path, cache_dir=None).get_db()

# decode synthetic data back to class codes
if onehot_layout:
    # one-hot / multinomial: collapse each class block by argmax (of the indicator for S4,
    # of the log-one-hot the multinomial reverse process ends on -- same operation).
    cat_k = (
        node_type_to_cat_num_classes
        if node_type_to_cat_num_classes is not None
        else {t: list(c.values()) for t, c in col_to_num_classes.items()}
    )
    node_type_to_X_syn = decode_onehot_synthetic_data(
        node_type_to_X_syn=node_type_to_X_syn,
        node_type_to_num_numerical_features=node_type_to_num_numerical_features,
        node_type_to_cat_num_classes=cat_k,
    )
else:
    node_type_to_X_syn = round_synthetic_data(
        node_type_to_X_syn=node_type_to_X_syn,
        node_type_to_X_real=extract_attributes(database_raw),
        node_type_to_num_numerical_features=node_type_to_num_numerical_features,
        categorical_decoding=args.categorical_decoding,
    )

if not generate_structure:
    database_syn = deepcopy(database_raw)  # only the structure will be kept

for table_name, table in database_syn.table_dict.items():
    df = table.df
    cols_no_id = [col for col in df.columns if not col.endswith("_id")]
    df[cols_no_id] = node_type_to_X_syn[table_name]

# for evaluation, save synthetic data to disk
database_syn_path = os.path.join(syn_data_dir, "synthetic_database")
write_database(database_syn, database_syn_path, suffix="_synthetic")

# also save real data because it needs to be sorted in the same was as synthetic data
database_real_path = os.path.join(workspace_dir, "real_database")
write_database(database_raw, database_real_path)

##################################
### 5. Data Quality Evaluation ###
##################################

# eval
print("Evaluating...")
eval_result = gen_multi_report(
    real_data_path=database_real_path,
    syn_data_path=database_syn_path,
    metadata_dir=data_path,
    eval_save_dir=syn_data_dir,
)

print(f"Final results on dataset {dataset_name}:")
print("Cardinality:", f'{eval_result["report"].get_properties()["Score"][2]*100:.04}')
print("Column Shapes:", f'{eval_result["report"].get_properties()["Score"][0]*100:.04}')
for i in range(len(eval_result["avg_scores"])):
    if i == 0:
        print("Intra-Table Trends:", f'{eval_result["avg_scores"][i]*100:.04}')
    else:
        print(f"{i}-HOP:", f'{eval_result["avg_scores"][i]*100:.04}')
print("AVG 2-WAY:", f'{eval_result["all_avg_score"]*100:.04}')

# wandb
wandb.log(
    {
        "eval/Cardinality": round(
            eval_result["report"].get_properties()["Score"][2] * 100, 2
        ),
        "eval/Column Shapes": round(
            eval_result["report"].get_properties()["Score"][0] * 100, 2
        ),
        "eval/AVG 2-WAY": round(eval_result["all_avg_score"] * 100, 2),
    }
)
for i in range(len(eval_result["avg_scores"])):
    if i == 0:
        wandb.log(
            {"eval/Intra-Table Trends": round(eval_result["avg_scores"][i] * 100, 2)}
        )
    else:
        wandb.log({f"eval/{i}-HOP": round(eval_result["avg_scores"][i] * 100, 2)})

print("Done!")
