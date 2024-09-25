import argparse
import os
from typing import List, Tuple

import polars as pl


def load_dataset(
    input_path: str, dataset_type: str
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    nodes_path = os.path.join(
        input_path, f"{dataset_type}_Dataset-1_nodes.parquet.gzip"
    )
    edges_path = os.path.join(
        input_path, f"{dataset_type}_Dataset-1_edges.parquet.gzip"
    )

    nodes_df = (
        pl.read_parquet(nodes_path)
        .unique("asm_hash")
        .with_columns(
            pl.concat_str([pl.col("graph_id"), pl.col("node_id")]).alias("id")
        )
    )

    edges_df = pl.read_parquet(edges_path)
    edges_df = edges_df.with_columns(
        edges_df["from"].cast(str).map_elements(lambda x: x[:-2], return_dtype=str),
        edges_df["to"].cast(str).map_elements(lambda x: x[:-2], return_dtype=str),
        pl.concat_str([pl.col("graph_id"), pl.col("from")]).alias("id_from"),
        pl.concat_str([pl.col("graph_id"), pl.col("to")]).alias("id_to"),
    )
    edges_df = (
        edges_df.join(nodes_df, left_on="id_from", right_on="id", suffix="_src")
        .join(nodes_df, left_on="id_to", right_on="id", suffix="_dst")[
            ["asm_hash", "asm_hash_dst"]
        ]
        .rename({"asm_hash": "from", "asm_hash_dst": "to"})
    )

    return nodes_df, edges_df


def filter_dataset(
    nodes_df: pl.DataFrame, edges_df: pl.DataFrame, other_nodes: List[pl.DataFrame]
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    filtered_nodes = nodes_df
    for other_df in other_nodes:
        filtered_nodes = filtered_nodes.join(other_df, how="anti", on="asm_hash")

    filtered_edges = edges_df.filter(
        (pl.col("from").is_in(filtered_nodes["asm_hash"]))
        & (pl.col("to").is_in(filtered_nodes["asm_hash"]))
    )

    return filtered_nodes, filtered_edges


def save_dataset(
    nodes_df: pl.DataFrame, edges_df: pl.DataFrame, output_path: str, dataset_type: str
) -> None:
    nodes_path = os.path.join(
        output_path, f"{dataset_type}_Dataset-1_nodes_filtered.parquet.gzip"
    )
    edges_path = os.path.join(
        output_path, f"{dataset_type}_Dataset-1_edges_filtered.parquet.gzip"
    )

    nodes_df.write_parquet(nodes_path)
    edges_df.write_parquet(edges_path)


def remove_common_nodes(input_path: str) -> None:
    print("Loading datasets...")
    train_nodes, train_edges = load_dataset(input_path, "training")
    val_nodes, val_edges = load_dataset(input_path, "validation")
    test_nodes, test_edges = load_dataset(input_path, "testing")

    print("Filtering datasets...")
    train_nodes_filtered, train_edges_filtered = filter_dataset(
        train_nodes, train_edges, [val_nodes, test_nodes]
    )
    val_nodes_filtered, val_edges_filtered = filter_dataset(
        val_nodes, val_edges, [train_nodes, test_nodes]
    )
    test_nodes_filtered, test_edges_filtered = filter_dataset(
        test_nodes, test_edges, [train_nodes, val_nodes]
    )

    print("Saving filtered datasets...")
    save_dataset(train_nodes_filtered, train_edges_filtered, input_path, "training")
    save_dataset(val_nodes_filtered, val_edges_filtered, input_path, "validation")
    save_dataset(test_nodes_filtered, test_edges_filtered, input_path, "testing")

    print("Done processing data")


def main():
    parser = argparse.ArgumentParser(
        description="Process the generated datasets removing common nodes across these one"
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path where the datasets are stored",
    )
    args = parser.parse_args()

    remove_common_nodes(args.input_path)


if __name__ == "__main__":
    main()
