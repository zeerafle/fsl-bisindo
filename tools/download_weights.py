import glob
import os
import tempfile
import zipfile
from urllib.request import urlretrieve

from omegaconf import OmegaConf

WEIGHTS_URL = {
    "AUTSL": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/autsl_slgcn.zip",
    "LSA64": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/lsa64_slgcn.zip",
    "CSL": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/csl_slgcn.zip",
}

METADATA_URL = {
    "AUTSL": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/autsl_metadata.zip",
    "LSA64": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/lsa64_metadata.zip",
    "CSL": "https://github.com/AI4Bharat/OpenHands/releases/download/checkpoints_v1/csl_metadata.zip",
}


for dataset, url in WEIGHTS_URL.items():
    # download weights zip
    filename = url.split("/")[-1]
    path = os.path.join(tempfile.gettempdir(), filename)
    urlretrieve(url, path)
    print(f"Downloaded {filename} for {dataset}")

    # extract zips
    with zipfile.ZipFile(path, "r") as zip_ref:
        zip_ref.extractall("./checkpoints/")
    print(f"Extracted {filename} for {dataset}")

    # download metadata zip
    metadata_filename = METADATA_URL[dataset].split("/")[-1]
    metadata_path = os.path.join(tempfile.gettempdir(), metadata_filename)
    urlretrieve(METADATA_URL[dataset], metadata_path)
    print(f"Downloaded {metadata_filename} for {dataset}")

    # extract metadata
    metadata_dir = f"./data/{dataset}"
    os.makedirs(metadata_dir, exist_ok=True)
    with zipfile.ZipFile(metadata_path, "r") as zip_ref:
        zip_ref.extractall(metadata_dir)
    print(f"Extracted {metadata_filename} for {dataset}")

    # load and modify config
    config_path = f"./checkpoints/{dataset.lower()}/sl_gcn/config.yaml"
    config = OmegaConf.load(config_path)

    # duplicate valid_pipeline as test_pipeline
    if hasattr(config, "data") and hasattr(config.data, "valid_pipeline"):
        config.data.test_pipeline = config.data.valid_pipeline

    # update paths to point to metadata directory
    # create dummy pose directories to trick openhands
    if dataset == "AUTSL":
        # create dummy directories
        train_pose_dir = f"{metadata_dir}/poses_pickle/train_poses/new_train_poses"
        test_pose_dir = f"{metadata_dir}/poses_pickle/test_poses/new_test_poses"
        os.makedirs(train_pose_dir, exist_ok=True)
        os.makedirs(test_pose_dir, exist_ok=True)

        # update config paths
        config.data.train_pipeline.dataset.split_file = (
            f"{metadata_dir}/AUTSL/train_labels.csv"
        )
        config.data.train_pipeline.dataset.root_dir = train_pose_dir
        config.data.train_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/AUTSL/SignList_ClassId_TR_EN.csv"
        )

        config.data.valid_pipeline.dataset.split_file = (
            f"{metadata_dir}/AUTSL/test_labels.csv"
        )
        config.data.valid_pipeline.dataset.root_dir = test_pose_dir
        config.data.valid_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/AUTSL/SignList_ClassId_TR_EN.csv"
        )

        config.data.test_pipeline.dataset.split_file = (
            f"{metadata_dir}/AUTSL/test_labels.csv"
        )
        config.data.test_pipeline.dataset.root_dir = test_pose_dir
        config.data.test_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/AUTSL/SignList_ClassId_TR_EN.csv"
        )

    elif dataset == "LSA64":
        # create dummy directory
        pose_dir = f"{metadata_dir}/lsa64/pose_all_cut"
        os.makedirs(pose_dir, exist_ok=True)

        # update config paths
        config.data.train_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/lsa64/lsa64_signs.md"
        )
        config.data.train_pipeline.dataset.root_dir = pose_dir

        config.data.valid_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/lsa64/lsa64_signs.md"
        )
        config.data.valid_pipeline.dataset.root_dir = pose_dir

        config.data.test_pipeline.dataset.class_mappings_file_path = (
            f"{metadata_dir}/lsa64/lsa64_signs.md"
        )
        config.data.test_pipeline.dataset.root_dir = pose_dir

    elif dataset == "CSL":
        # create dummy directory
        pose_dir = f"{metadata_dir}/CSL/word/pose_mediapipe"
        os.makedirs(pose_dir, exist_ok=True)

        # update config paths
        config.data.train_pipeline.dataset.split_file = (
            f"{metadata_dir}/CSL/word/gloss_label.txt"
        )
        config.data.train_pipeline.dataset.root_dir = pose_dir

        config.data.valid_pipeline.dataset.split_file = (
            f"{metadata_dir}/CSL/word/gloss_label.txt"
        )
        config.data.valid_pipeline.dataset.root_dir = pose_dir

        config.data.test_pipeline.dataset.split_file = (
            f"{metadata_dir}/CSL/word/gloss_label.txt"
        )
        config.data.test_pipeline.dataset.root_dir = pose_dir

    # add name key
    config.name = f"{dataset.lower()}_slgcn"

    # add pretrained key with checkpoint path as value
    checkpoint_path = glob.glob(f"./checkpoints/{dataset.lower()}/sl_gcn/*.ckpt")[0]
    config.pretrained = checkpoint_path

    # change params.graph_args.num_points to num_nodes, if any
    if hasattr(config.model.encoder.params.graph_args, "num_points"):
        config.model.encoder.params.graph_args.num_nodes = (
            config.model.encoder.params.graph_args.num_points
        )
        del config.model.encoder.params.graph_args.num_points

    # save modified config to configs folder
    modified_config_path = f"./configs/backbones/{dataset.lower()}_slgcn.yaml"
    OmegaConf.save(config, modified_config_path)
    print(f"Saved modified config to {modified_config_path}")
