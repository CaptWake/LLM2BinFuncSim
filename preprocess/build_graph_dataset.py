import base64
import json
import multiprocessing
from hashlib import sha256
from pathlib import Path

import click
import lief
import pandas as pd
import pypcode
from tqdm import tqdm


def retrieve_fname_from_idb_path(idb_path: str) -> str:
    return "/".join(idb_path.split("/")[3:])[:-4] + "_acfg_disasm.json"


def get_ghidra_language_id(binary_path):
    try:
        binary = lief.parse(binary_path)

        # Determine architecture
        machine_type = binary.header.machine_type
        if machine_type == lief.ELF.ARCH.X86_64:
            arch = "x86"
            bits = 64
            extension = "default"
        elif machine_type == lief.ELF.ARCH.I386:
            arch = "x86"
            bits = 32
            extension = "default"
        elif machine_type == lief.ELF.ARCH.ARM:
            arch = "ARM"
            bits = 32
            extension = "v8"
        elif machine_type == lief.ELF.ARCH.AARCH64:
            arch = "AARCH64"
            bits = 64
            extension = "v8A"
        elif machine_type == lief.ELF.ARCH.MIPS:
            arch = "MIPS"
            bits = (
                32
                if binary.header.identity_class == lief.ELF.Header.CLASS.ELF32
                else 64
            )
            extension = "default"

        # Determine endianness
        endian = (
            "LE"
            if binary.header.identity_data == lief.ELF.Header.ELF_DATA.LSB
            else "BE"
        )

        # Construct Ghidra-like language ID
        language_id = f"{arch}:{endian}:{bits}:{extension}"

        return language_id

    except Exception as e:
        return f"Error parsing file {binary_path}: {str(e)}"


def format_varnode(vn):
    if vn.space.name == "register":
        return f"(register, 0x{vn.offset:x}, {vn.size})"
    elif vn.space.name == "const":
        return f"(const, 0x{vn.offset:x}, {vn.size})"
    elif vn.space.name == "unique":
        return f"(unique, 0x{vn.offset:x}, {vn.size})"
    else:
        return f"({vn.space.name}, 0x{vn.offset:x}, {vn.size})"


def format_pcode_op(op) -> str:
    # Format the output varnode
    if op.output:
        out_str = format_varnode(op.output)
    else:
        out_str = ""

    # Format the input varnodes
    in_strs = [format_varnode(vn) for vn in op.inputs]

    # Combine into the desired format
    if out_str:
        return f"{out_str} = {op.opcode.name} {', '.join(in_strs)}"
    else:
        return f"{op.opcode.name} {', '.join(in_strs)}"


def pcode_from_bytecode(bytecode, binary_path):
    # Define architecture for pypcode
    pcode_arch = get_ghidra_language_id(binary_path)

    try:
        # Create a pypcode context
        ctx = pypcode.Context(pcode_arch)
        # Translate the bytecode to P-code
        pcode_ops = ctx.translate(bytecode)

        return [format_pcode_op(op) for op in pcode_ops.ops]

    except Exception as e:
        print(f"Error generating pcode: {e}")
        print(pcode_arch, binary_path)


def parse_nxopr(pcode_asm):
    s = pcode_asm.find("(")
    e = pcode_asm.find(")")
    return tuple(pcode_asm[s + 1 : e].split(", ")), pcode_asm[e + 1 :].strip()


def parse_pcode(pcode_asm):
    """
    Examples:
    (register, 0x20, 4) COPY (const, 0x0, 4)
    (unique, 0x8380, 4) INT_ADD (register, 0x4c, 4) , (const, 0xfffffff0, 4)
     ---  STORE (STORE, 0x1a1, 0) , (unique, 0x8280, 4) , (register, 0x20, 4)
     ---  BRANCH (ram, 0x22128, 4)
    """
    NOP_OPERAND = " --- "
    dst_opr = None
    if pcode_asm.startswith(NOP_OPERAND):
        pcode_asm = pcode_asm[len(NOP_OPERAND) + 1 :]
    else:
        dst_opr, pcode_asm = parse_nxopr(pcode_asm)
    opc_e = pcode_asm.find(" ")
    if opc_e != -1:
        opc, pcode_asm = pcode_asm[:opc_e], pcode_asm[opc_e:].strip()
    else:
        opc, pcode_asm = pcode_asm, ""
    oprs = (
        []
        if dst_opr is None
        else [
            dst_opr,
        ]
    )
    while len(pcode_asm) != 0:
        src_opr, pcode_asm = parse_nxopr(pcode_asm)
        oprs.append(src_opr)
    return (opc, oprs)


def normalize_pcode_opr(opr, arch):
    if opr[0] in ["register"]:
        return f"{arch}_reg", arch + "_" + "_".join(opr)
    elif opr[0] in ["STORE", "const"]:
        return "val", "_".join(opr[:-1])  # omit dummy size field
    elif opr[0] in ["unique", "NewUnique", "ram", "stack", "VARIABLE"]:
        return "val", opr[0]
    else:
        raise Exception(f"Unkown operand type {opr[0]}. FULL: {opr}. ")


def normalize_pcode(pcode, arch):
    normalized_pcode = [("opc", pcode[0])]
    for opr in pcode[1]:
        normalized_pcode.append(normalize_pcode_opr(opr, arch))
    return normalized_pcode


def get_graph_info(fname, idb_path, fva, mode):
    # Construct the base command list
    data = None
    with open(fname, "r") as f:
        data = json.load(f)

    base_dict = data[idb_path][fva]
    bbs_dict = base_dict["basic_blocks"]

    nodes = []

    for bb in sorted(bbs_dict):
        try:
            if mode == "pcode":
                bytecode = base64.b64decode(bbs_dict[bb]["b64_bytes"])
                pcode = pcode_from_bytecode(
                    bytecode,
                    "data/binaries" + idb_path.split(".i64")[0].split("IDBs")[1],
                )
                # pcode = "\n".join(pcode) + "\n"
                pcode_hash = sha256(pcode.encode()).hexdigest()
                nodes.append([bb, pcode_hash, pcode])
            else:
                asm_code = "\n".join(bbs_dict[bb]["bb_disasm"]) + "\n"
                asm_hash = sha256(asm_code.encode()).hexdigest()
                nodes.append([bb, asm_hash, asm_code])
        except:
            continue
    return nodes, base_dict["edges"]


# idb_path,fva,func_name,start_ea,end_ea,bb_num,hashopcodes,project,library,arch,bit,compiler,version,optimizations,asm_code
def process_binary_block(binary_block, disasm_path, mode):
    node_info = {
        "graph_id": [],
        "node_id": [],
        "asm_hash": [],
        "asm_code": [],
    }
    edge_lists = {"graph_id": [], "from": [], "to": []}
    graph_info = {
        "graph_id": [],
        "idb_path": [],
        "func_name": [],
        "fva": [],
        "arch": [],
        "compiler": [],
        "version": [],
        "optimizations": [],
    }

    for _, binary_info in binary_block.iterrows():
        fname = retrieve_fname_from_idb_path(binary_info["idb_path"])

        nodes, edge_list = get_graph_info(
            disasm_path + fname, binary_info["idb_path"], binary_info["fva"], mode
        )

        graph_id = sha256(
            (
                binary_info["idb_path"] + binary_info["func_name"] + binary_info["fva"]
            ).encode()
        ).hexdigest()
        graph_info["graph_id"].append(graph_id)
        graph_info["idb_path"].append(binary_info["idb_path"])
        graph_info["func_name"].append(binary_info["func_name"])
        graph_info["fva"].append(binary_info["fva"])
        graph_info["arch"].append(binary_info["arch"] + str(binary_info["bit"]))
        graph_info["compiler"].append(binary_info["compiler"])
        graph_info["version"].append(binary_info["version"])
        graph_info["optimizations"].append(binary_info["optimizations"])

        node_info["graph_id"].extend([graph_id for i in range(len(nodes))])
        node_info["node_id"].extend([node[0] for node in nodes])
        node_info["asm_hash"].extend([node[1] for node in nodes])
        node_info["asm_code"].extend([node[2] for node in nodes])

        edge_lists["graph_id"].extend([graph_id for i in range(len(edge_list))])
        edge_lists["from"].extend([int(pair[0]) for pair in edge_list])
        edge_lists["to"].extend([int(pair[1]) for pair in edge_list])

    return graph_info, node_info, edge_lists


def process_dataset(
    input_dataset, output_dataset, disasm_path, sample_frac, seed, mode
):
    df = pd.read_csv(input_dataset, index_col=0).sample(
        frac=sample_frac, random_state=seed
    )
    df_keys = pd.DataFrame({"graph_id": [], "idb_path": [], "func_name": [], "fva": []})
    df_nodes = pd.DataFrame({"graph_id": [], "node_id": [], "asm_code": []})
    df_edges = pd.DataFrame({"graph_id": [], "from": [], "to": []})

    num_cores = multiprocessing.cpu_count()
    block_size = len(df) // num_cores  # Assign block size based on available cores

    binary_blocks = [df.iloc[i : i + block_size] for i in range(0, len(df), block_size)]

    with multiprocessing.Pool(processes=num_cores) as pool:
        results = list(
            tqdm(
                pool.starmap(
                    process_binary_block,
                    [
                        (binary_block, disasm_path, mode)
                        for binary_block in binary_blocks
                    ],
                ),
                total=len(binary_blocks),
            )
        )

    print(f"[*] Saving Results in {output_dataset}_[nodes|edges].parquet")
    for graph_ids, nodes_info, edge_list in tqdm(results):
        df_keys = pd.concat(
            [df_keys, pd.DataFrame(graph_ids)], axis=0, ignore_index=True
        )
        df_nodes = pd.concat(
            [df_nodes, pd.DataFrame(nodes_info)], axis=0, ignore_index=True
        )
        df_edges = pd.concat(
            [df_edges, pd.DataFrame(edge_list)], axis=0, ignore_index=True
        )

    df_keys.to_parquet(str(output_dataset) + "_keys.parquet", index=False)
    df_nodes.to_parquet(str(output_dataset) + "_nodes.parquet", index=False)
    df_edges.to_parquet(str(output_dataset) + "_edges.parquet", index=False)


@click.command()
@click.option("--pcode", is_flag=True, help="Generate P-code instead of raw assembly")
@click.option("--subsample", is_flag=True, help="Subsample the datasets")
@click.option("--seed", default=42, type=int, help="Random seed for sampling")
def main(pcode, subsample, seed):
    input_dataset_paths = [
        "./HermesSim/data/Dataset-1/training_Dataset-1.csv",
        "./HermesSim/data/Dataset-1/testing_Dataset-1.csv",
        "./HermesSim/data/Dataset-1/validation_Dataset-1.csv",
    ]

    mode = "pcode" if pcode else "raw"

    output_dataset_paths = [
        f"./HermesSim/data/Dataset-1/training_Dataset-1",
        f"./HermesSim/data/Dataset-1/testing_Dataset-1",
        f"./HermesSim/data/Dataset-1/validation_Dataset-1",
    ]

    disasm_paths = [
        "./HermesSim/data/Dataset-1/raw/features/training/acfg_disasm_Dataset-1_training/",
        "./HermesSim/data/Dataset-1/raw/features/testing/acfg_disasm_Dataset-1_testing/",
        "./HermesSim/data/Dataset-1/raw/features/validation/acfg_disasm_Dataset-1_validation/",
    ]

    sample_fracs = [0.01, 0.01, 0.1] if subsample else [1.0, 1.0, 1.0]

    for _, (in_ds_path, out_ds_path, disasm_path, sample_frac) in enumerate(
        zip(input_dataset_paths, output_dataset_paths, disasm_paths, sample_fracs)
    ):
        print(f"[*] Processing Dataset {in_ds_path}")
        process_dataset(
            Path(in_ds_path), Path(out_ds_path), disasm_path, sample_frac, seed, mode
        )
        print(f"[*] Saved Results in {out_ds_path}_[nodes|edges|keys].parquet")

    print(f"[*] Done processing data")


if __name__ == "__main__":
    main()
