import json
import os
import pickle
from collections import defaultdict
from typing import Optional, TypeAlias

import click
import networkx as nx
import numpy as np
import polars as pl
from polars import DataFrame, LazyFrame
from pydantic import BaseModel, Field
from scipy.sparse import coo_matrix
from tqdm import tqdm

# Constants
CLS_SIZE: int = 768
GRAPH_TYPE: str = "ACFG"


# Type aliases
EmbeddingsDict: TypeAlias = dict[str, np.ndarray]
GraphPacked: TypeAlias = tuple[coo_matrix, list[int]]
ProcessResult: TypeAlias = tuple[str, dict, dict]


class Config(BaseModel):
    input_dir: str = Field(..., description="Input directory path")
    output_dir: str = Field(..., description="Output directory path")
    features_filename: str = Field(..., description="Name of the features file")
    dataset: str = Field(..., description="Dataset name")
    experiment: str = Field(..., description="Experiment name")
    out_format: str = Field(..., description="Output format (pkl or json)")
    embeddings_path: str = Field(..., description="Path to embeddings")
    nodes_path: str = Field(..., description="Path to nodes file")
    keys_path: str = Field(..., description="Path to keys file")
    dump_str: bool = Field(False, description="Whether to dump string representation")
    dump_pkl: bool = Field(True, description="Whether to dump pickle representation")


def load_config(config_path: str) -> Config:
    with open(config_path, "r") as f:
        return Config(**json.load(f))


def create_graph_embeddings(fva_data: dict) -> GraphPacked:
    nodes: list[int] = fva_data["nodes"]
    edges: list[tuple[int, int]] = fva_data["edges"]
    G = nx.MultiDiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    nodelist = list(G.nodes())
    adj_mat: coo_matrix = nx.to_scipy_sparse_array(
        G, nodelist=nodelist, dtype=np.int8, format="coo"
    )
    return adj_mat, nodelist


def coo2tuple(
    coo_mat: coo_matrix,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    return (coo_mat.row, coo_mat.col, coo_mat.data, *coo_mat.shape)


def load_embeddings(node_list: list[int], emb_dict: EmbeddingsDict) -> np.ndarray:
    embeddings: np.ndarray = np.zeros(
        shape=(len(node_list), CLS_SIZE), dtype=np.float32
    )
    for i, node in enumerate(node_list):
        embeddings[i] = emb_dict.get(str(node), np.zeros(CLS_SIZE))
    return embeddings


def coo_matrix_to_str(cmat: coo_matrix) -> str:
    components: list[str] = [
        ";".join(map(str, arr)) for arr in (cmat.row, cmat.col, cmat.data)
    ]
    components.extend(map(str, cmat.shape))
    return "::".join(components)


def process_one_file(args: tuple[str, bool, bool, DataFrame]) -> ProcessResult:
    json_path: str
    dump_str: bool
    dump_pkl: bool
    emb_dict: DataFrame
    json_path, dump_str, dump_pkl, emb_dict = args

    with open(json_path) as f_in:
        jj = json.load(f_in)

    idb_path: str = list(jj.keys())[0]
    j_data: dict = jj[idb_path]

    str_func_dict = defaultdict(dict) if dump_str else {}
    pkl_func_dict = defaultdict(dict) if dump_pkl else {}

    for fva, fva_df in emb_dict.filter(pl.col("idb_path") == idb_path).group_by("fva"):
        fva = fva[0]
        g_coo_mat, nodes = create_graph_embeddings(j_data[fva])
        node_emb_dict: EmbeddingsDict = dict(
            fva_df.select(["node_id", "embeddings"]).iter_rows()
        )
        f_list = load_embeddings(nodes, node_emb_dict)

        assert isinstance(fva, str)

        if not fva.startswith("0x"):
            fva = hex(int(fva, 10))

        if len(f_list) > 0:
            data = {
                "graph": (
                    coo_matrix_to_str(g_coo_mat) if dump_str else coo2tuple(g_coo_mat)
                ),
                "bb_embeddings": f_list,
            }
            if dump_str:
                str_func_dict[GRAPH_TYPE][fva] = data
            if dump_pkl:
                pkl_func_dict[GRAPH_TYPE][fva] = data

    return idb_path, str_func_dict, pkl_func_dict


def create_functions_dict(config: Config, emb_dict: DataFrame) -> tuple[dict, dict]:
    args: list[tuple[str, bool, bool, DataFrame]] = [
        (
            os.path.join(
                config.input_dir,
                idb_path[0].split("/")[-1].replace(".i64", "_acfg_disasm.json"),
            ),
            config.dump_str,
            config.dump_pkl,
            emb_df,
        )
        for idb_path, emb_df in emb_dict.group_by("idb_path")
    ]

    results: list[ProcessResult] = list(
        tqdm(map(process_one_file, args), total=len(args))
    )

    str_func_dict = defaultdict(lambda: defaultdict(dict))
    pkl_func_dict = defaultdict(lambda: defaultdict(dict))

    for idb_path, str_func_one, pkl_func_one in results:
        for gtype in [GRAPH_TYPE]:
            if config.dump_str:
                str_func_dict[gtype][idb_path] = str_func_one[gtype]
            if config.dump_pkl:
                pkl_func_dict[gtype][idb_path] = pkl_func_one[gtype]

    return str_func_dict, pkl_func_dict


def get_sub_dir(output_dir: str, experiment: str, dataset: str) -> str:
    sub_dir: str = os.path.join(output_dir, experiment, dataset)
    os.makedirs(sub_dir, exist_ok=True)
    return sub_dir


def load_data(config: Config, partition: str) -> DataFrame:
    emb_dict: LazyFrame = pl.scan_parquet(config.embeddings_path)
    nodes_df: LazyFrame = pl.scan_parquet(config.nodes_path).select(
        ["graph_id", "asm_hash", "node_id"]
    )
    keys_df: LazyFrame = pl.scan_parquet(config.keys_path).select(
        ["graph_id", "idb_path", "fva"]
    )

    if partition != "training":
        emb_dict = emb_dict.select(["asm_hash", "embeddings"])
        if partition == "testing":
            emb_dict = emb_dict.unique("asm_hash")

    return (
        emb_dict.join(nodes_df, on="asm_hash")
        .join(keys_df, on="graph_id")
        .select("idb_path", "fva", "node_id", "embeddings")
        .collect()
    )


@click.command()
@click.option("-c", "--config", required=True, help="Path to the configuration file")
@click.option(
    "-p",
    "--partition",
    required=True,
    help='Partition type ("training", "validation" or "testing")',
)
def main(config: str, partition: str) -> None:
    config_data: Config = load_config(config)
    os.makedirs(config_data.output_dir, exist_ok=True)

    emb_dict: DataFrame = load_data(config_data, partition)
    str_dict, pkl_dict = create_functions_dict(config_data, emb_dict)

    output_dir: str = get_sub_dir(
        config_data.output_dir,
        config_data.experiment,
        f"{config_data.dataset}_{partition}",
    )

    for gtype, func_dict in [(GRAPH_TYPE, str_dict), (GRAPH_TYPE, pkl_dict)]:
        if func_dict:
            output_path: str = os.path.join(
                output_dir, f"{config_data.features_filename}"
            )
            print(f"Saving results to {output_path}")

            if isinstance(func_dict, defaultdict):
                func_dict = dict(func_dict[gtype])

            with open(
                f"{output_path}.{'json' if gtype == str_dict else 'pkl'}",
                "w" if gtype == str_dict else "wb",
            ) as f_out:
                if gtype == str_dict:
                    json.dump(func_dict, f_out)
                else:
                    pickle.dump(func_dict, f_out)


if __name__ == "__main__":
    main()
