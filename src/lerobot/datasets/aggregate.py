#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import shutil
from pathlib import Path

import pandas as pd
import tqdm

from lerobot.datasets.compute_stats import aggregate_stats
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.datasets.utils import (
    DEFAULT_CHUNK_SIZE,
    DEFAULT_DATA_FILE_SIZE_IN_MB,
    DEFAULT_DATA_PATH,
    DEFAULT_EPISODES_PATH,
    DEFAULT_VIDEO_FILE_SIZE_IN_MB,
    DEFAULT_VIDEO_PATH,
    get_file_size_in_mb,
    get_parquet_file_size_in_mb,
    to_parquet_with_hf_images,
    update_chunk_file_indices,
    write_info,
    write_stats,
    write_tasks,
)
from lerobot.datasets.video_utils import concatenate_video_files, get_video_duration_in_s


def _metadata_label(meta: LeRobotDatasetMetadata) -> str:
    """生成简洁的数据集标识，方便从日志定位是哪一个源数据集不一致。"""
    return f"repo_id={meta.repo_id}, root={meta.root}"


def _summarize_features(features: dict) -> str:
    """汇总 feature 名称、dtype 和 shape，用于合并前的 metadata 诊断日志。"""
    parts = []
    for name, feature in sorted(features.items()):
        dtype = feature.get("dtype")
        shape = feature.get("shape")
        parts.append(f"{name}(dtype={dtype}, shape={shape})")
    return "[" + ", ".join(parts) + "]"


def _log_feature_mismatch(base_features: dict, current_features: dict) -> None:
    """在抛出 features 不一致错误前，先打印缺失、额外和定义变化的字段。"""
    base_keys = set(base_features)
    current_keys = set(current_features)
    missing_keys = sorted(base_keys - current_keys)
    extra_keys = sorted(current_keys - base_keys)
    changed_keys = sorted(
        key for key in base_keys & current_keys if base_features[key] != current_features[key]
    )

    if missing_keys:
        logging.error("features 缺失字段: %s", missing_keys)
    if extra_keys:
        logging.error("features 额外字段: %s", extra_keys)
    if changed_keys:
        for key in changed_keys:
            logging.error(
                "features 字段定义不一致: key=%s, expected=%s, got=%s",
                key,
                base_features[key],
                current_features[key],
            )


def validate_all_metadata(all_metadata: list[LeRobotDatasetMetadata]):
    """Validates that all dataset metadata have consistent properties.

    Ensures all datasets have the same fps, robot_type, and features to guarantee
    compatibility when aggregating them into a single dataset.

    Args:
        all_metadata: List of LeRobotDatasetMetadata objects to validate.

    Returns:
        tuple: A tuple containing (fps, robot_type, features) from the first metadata.

    Raises:
        ValueError: If any metadata has different fps, robot_type, or features
                   than the first metadata in the list.
    """

    fps = all_metadata[0].fps
    robot_type = all_metadata[0].robot_type
    features = all_metadata[0].features
    logging.info(
        "metadata 基准: %s, fps=%s, robot_type=%s, total_features=%d, features=%s",
        _metadata_label(all_metadata[0]),
        fps,
        robot_type,
        len(features),
        _summarize_features(features),
    )

    for meta in tqdm.tqdm(all_metadata, desc="Validate all meta data"):
        logging.info(
            "metadata 检查: %s, fps=%s, robot_type=%s, total_features=%d",
            _metadata_label(meta),
            meta.fps,
            meta.robot_type,
            len(meta.features),
        )
        if fps != meta.fps:
            logging.error(
                "metadata 不一致: field=fps, dataset=%s, expected=%s, got=%s",
                _metadata_label(meta),
                fps,
                meta.fps,
            )
            raise ValueError(f"Same fps is expected, but got fps={meta.fps} instead of {fps}.")
        if robot_type != meta.robot_type:
            logging.error(
                "metadata 不一致: field=robot_type, dataset=%s, expected=%s, got=%s",
                _metadata_label(meta),
                robot_type,
                meta.robot_type,
            )
            raise ValueError(
                f"Same robot_type is expected, but got robot_type={meta.robot_type} instead of {robot_type}."
            )
        if features != meta.features:
            logging.error(
                "metadata 不一致: field=features, dataset=%s, expected_total=%d, got_total=%d",
                _metadata_label(meta),
                len(features),
                len(meta.features),
            )
            _log_feature_mismatch(features, meta.features)
            raise ValueError(
                f"Same features is expected, but got features={meta.features} instead of {features}."
            )

    return fps, robot_type, features


def update_data_df(df, src_meta, dst_meta):
    """Updates a data DataFrame with new indices and task mappings for aggregation.

    Adjusts episode indices, frame indices, and task indices to account for
    previously aggregated data in the destination dataset.

    Args:
        df: DataFrame containing the data to be updated.
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices.
    """

    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]
    df["index"] = df["index"] + dst_meta.info["total_frames"]

    src_task_names = src_meta.tasks.index.take(df["task_index"].to_numpy())
    df["task_index"] = dst_meta.tasks.loc[src_task_names, "task_index"].to_numpy()

    return df


def update_meta_data(
    df,
    dst_meta,
    meta_idx,
    data_idx,
    videos_idx,
):
    """Updates metadata DataFrame with new chunk, file, and timestamp indices.

    Adjusts all indices and timestamps to account for previously aggregated
    data and videos in the destination dataset.

    Args:
        df: DataFrame containing the metadata to be updated.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary containing current metadata chunk and file indices.
        data_idx: Dictionary containing current data chunk and file indices.
        videos_idx: Dictionary containing current video indices and timestamps.

    Returns:
        pd.DataFrame: Updated DataFrame with adjusted indices and timestamps.
    """

    df["meta/episodes/chunk_index"] = df["meta/episodes/chunk_index"] + meta_idx["chunk"]
    df["meta/episodes/file_index"] = df["meta/episodes/file_index"] + meta_idx["file"]
    df["data/chunk_index"] = df["data/chunk_index"] + data_idx["chunk"]
    df["data/file_index"] = df["data/file_index"] + data_idx["file"]
    for key, video_idx in videos_idx.items():
        orig_chunk_col = f"videos/{key}/chunk_index"
        orig_file_col = f"videos/{key}/file_index"
        df["_orig_chunk"] = df[orig_chunk_col].copy()
        df["_orig_file"] = df[orig_file_col].copy()

        # 按源视频文件映射到真实的目标视频文件，并应用该目标文件内的时间偏移。
        # 不能把整个 source dataset 的 metadata 都写成最后一个目标 file，否则会出现
        # episode 指向 file-025.mp4，却使用其它目标文件累计时间戳的越界问题。
        src_to_offset = video_idx.get("src_to_offset", {})
        src_to_dst = video_idx.get("src_to_dst", {})
        if src_to_offset:
            missing_mappings = set()
            for idx in df.index:
                src_key = (df.at[idx, "_orig_chunk"], df.at[idx, "_orig_file"])
                if src_key not in src_to_dst:
                    missing_mappings.add(src_key)
                dst_chunk, dst_file = src_to_dst.get(src_key, (video_idx["chunk"], video_idx["file"]))
                offset = src_to_offset.get(src_key, 0)
                df.at[idx, orig_chunk_col] = dst_chunk
                df.at[idx, orig_file_col] = dst_file
                df.at[idx, f"videos/{key}/from_timestamp"] += offset
                df.at[idx, f"videos/{key}/to_timestamp"] += offset
            logging.info(
                "视频 metadata 映射完成: key=%s, source_files=%d, rows=%d",
                key,
                len(src_to_dst),
                len(df),
            )
            if missing_mappings:
                logging.warning(
                    "视频 metadata 存在缺失映射: key=%s, missing=%s, fallback_dst=(chunk-%03d,file-%03d)",
                    key,
                    sorted(missing_mappings),
                    video_idx["chunk"],
                    video_idx["file"],
                )
        else:
            # 兼容旧调用路径：没有 per-source-file 映射时才整体指向当前目标文件。
            df[orig_chunk_col] = video_idx["chunk"]
            df[orig_file_col] = video_idx["file"]
            df[f"videos/{key}/from_timestamp"] = (
                df[f"videos/{key}/from_timestamp"] + video_idx["latest_duration"]
            )
            df[f"videos/{key}/to_timestamp"] = df[f"videos/{key}/to_timestamp"] + video_idx["latest_duration"]
            logging.info(
                "视频 metadata 使用旧式整体映射: key=%s, dst=(chunk-%03d,file-%03d), offset=%.6f",
                key,
                video_idx["chunk"],
                video_idx["file"],
                video_idx["latest_duration"],
            )

        # Clean up temporary columns
        df = df.drop(columns=["_orig_chunk", "_orig_file"])

    df["dataset_from_index"] = df["dataset_from_index"] + dst_meta.info["total_frames"]
    df["dataset_to_index"] = df["dataset_to_index"] + dst_meta.info["total_frames"]
    df["episode_index"] = df["episode_index"] + dst_meta.info["total_episodes"]

    return df


def aggregate_datasets(
    repo_ids: list[str],
    aggr_repo_id: str,
    roots: list[Path] | None = None,
    aggr_root: Path | None = None,
    data_files_size_in_mb: float | None = None,
    video_files_size_in_mb: float | None = None,
    chunk_size: int | None = None,
):
    """Aggregates multiple LeRobot datasets into a single unified dataset.

    This is the main function that orchestrates the aggregation process by:
    1. Loading and validating all source dataset metadata
    2. Creating a new destination dataset with unified tasks
    3. Aggregating videos, data, and metadata from all source datasets
    4. Finalizing the aggregated dataset with proper statistics

    Args:
        repo_ids: List of repository IDs for the datasets to aggregate.
        aggr_repo_id: Repository ID for the aggregated output dataset.
        roots: Optional list of root paths for the source datasets.
        aggr_root: Optional root path for the aggregated dataset.
        data_files_size_in_mb: Maximum size for data files in MB (defaults to DEFAULT_DATA_FILE_SIZE_IN_MB)
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)
    """
    logging.info("Start aggregate_datasets")

    if data_files_size_in_mb is None:
        data_files_size_in_mb = DEFAULT_DATA_FILE_SIZE_IN_MB
    if video_files_size_in_mb is None:
        video_files_size_in_mb = DEFAULT_VIDEO_FILE_SIZE_IN_MB
    if chunk_size is None:
        chunk_size = DEFAULT_CHUNK_SIZE

    all_metadata = (
        [LeRobotDatasetMetadata(repo_id) for repo_id in repo_ids]
        if roots is None
        else [
            LeRobotDatasetMetadata(repo_id, root=root) for repo_id, root in zip(repo_ids, roots, strict=False)
        ]
    )
    fps, robot_type, features = validate_all_metadata(all_metadata)
    video_keys = [key for key in features if features[key]["dtype"] == "video"]

    dst_meta = LeRobotDatasetMetadata.create(
        repo_id=aggr_repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        root=aggr_root,
        use_videos=len(video_keys) > 0,
        chunks_size=chunk_size,
        data_files_size_in_mb=data_files_size_in_mb,
        video_files_size_in_mb=video_files_size_in_mb,
    )

    logging.info("Find all tasks")
    unique_tasks = pd.concat([m.tasks for m in all_metadata]).index.unique()
    dst_meta.tasks = pd.DataFrame({"task_index": range(len(unique_tasks))}, index=unique_tasks)

    meta_idx = {"chunk": 0, "file": 0}
    data_idx = {"chunk": 0, "file": 0}
    # current_duration 表示“当前目标 mp4 文件内”已经累计的时长；
    # latest_duration 保留为全局累计时长，仅用于兼容旧的 fallback 逻辑。
    videos_idx = {
        key: {"chunk": 0, "file": 0, "current_duration": 0, "latest_duration": 0, "episode_duration": 0}
        for key in video_keys
    }

    dst_meta.episodes = {}

    for src_meta in tqdm.tqdm(all_metadata, desc="Copy data and videos"):
        videos_idx = aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size)
        data_idx = aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size)

        meta_idx = aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx)

        dst_meta.info["total_episodes"] += src_meta.total_episodes
        dst_meta.info["total_frames"] += src_meta.total_frames

    finalize_aggregation(dst_meta, all_metadata)
    logging.info("Aggregation complete.")


def aggregate_videos(src_meta, dst_meta, videos_idx, video_files_size_in_mb, chunk_size):
    """Aggregates video chunks from a source dataset into the destination dataset.

    Handles video file concatenation and rotation based on file size limits.
    Creates new video files when size limits are exceeded.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        videos_idx: Dictionary tracking video chunk and file indices.
        video_files_size_in_mb: Maximum size for video files in MB (defaults to DEFAULT_VIDEO_FILE_SIZE_IN_MB)
        chunk_size: Maximum number of files per chunk (defaults to DEFAULT_CHUNK_SIZE)

    Returns:
        dict: Updated videos_idx with current chunk and file indices.
    """
    for key in videos_idx:
        videos_idx[key]["episode_duration"] = 0
        # 记录每个源视频文件在目标视频文件中的时间偏移，后续写 episode metadata 时会用到。
        videos_idx[key]["src_to_offset"] = {}
        # 记录每个源视频文件最终写入的目标视频文件，避免 metadata 全部指向最后一个目标文件。
        videos_idx[key]["src_to_dst"] = {}

    for key, video_idx in videos_idx.items():
        unique_chunk_file_pairs = {
            (chunk, file)
            for chunk, file in zip(
                src_meta.episodes[f"videos/{key}/chunk_index"],
                src_meta.episodes[f"videos/{key}/file_index"],
                strict=False,
            )
        }
        unique_chunk_file_pairs = sorted(unique_chunk_file_pairs)

        chunk_idx = video_idx["chunk"]
        file_idx = video_idx["file"]
        current_offset = video_idx["current_duration"]
        logging.info(
            "开始合并视频流: key=%s, source_files=%d, start_dst=(chunk-%03d,file-%03d), current_duration=%.6f",
            key,
            len(unique_chunk_file_pairs),
            chunk_idx,
            file_idx,
            current_offset,
        )

        for src_chunk_idx, src_file_idx in unique_chunk_file_pairs:
            src_path = src_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=src_chunk_idx,
                file_index=src_file_idx,
            )

            dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                video_key=key,
                chunk_index=chunk_idx,
                file_index=file_idx,
            )

            src_duration = get_video_duration_in_s(src_path)

            if not dst_path.exists():
                # 目标文件刚创建时，源文件起点就是当前目标文件内的累计偏移。
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                logging.info(
                    "创建目标视频并复制: key=%s, src=(chunk-%03d,file-%03d), dst=(chunk-%03d,file-%03d), "
                    "offset=%.6f, src_duration=%.6f",
                    key,
                    src_chunk_idx,
                    src_file_idx,
                    chunk_idx,
                    file_idx,
                    current_offset,
                    src_duration,
                )
                videos_idx[key]["episode_duration"] += src_duration
                current_offset += src_duration
                continue

            # 追加前先检查目标文件大小，超过阈值就轮转到新的 mp4。
            src_size = get_file_size_in_mb(src_path)
            dst_size = get_file_size_in_mb(dst_path)

            if dst_size + src_size >= video_files_size_in_mb:
                # 轮转到新的目标 mp4 后，源文件从新文件 0 秒开始，不能再沿用全局累计时长。
                old_chunk_idx, old_file_idx = chunk_idx, file_idx
                chunk_idx, file_idx = update_chunk_file_indices(chunk_idx, file_idx, chunk_size)
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = 0
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                dst_path = dst_meta.root / DEFAULT_VIDEO_PATH.format(
                    video_key=key,
                    chunk_index=chunk_idx,
                    file_index=file_idx,
                )
                dst_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(str(src_path), str(dst_path))
                logging.info(
                    "目标视频轮转并复制: key=%s, src=(chunk-%03d,file-%03d), "
                    "old_dst=(chunk-%03d,file-%03d), new_dst=(chunk-%03d,file-%03d), "
                    "dst_size_mb=%.3f, src_size_mb=%.3f, limit_mb=%.3f, offset=0.000000, src_duration=%.6f",
                    key,
                    src_chunk_idx,
                    src_file_idx,
                    old_chunk_idx,
                    old_file_idx,
                    chunk_idx,
                    file_idx,
                    dst_size,
                    src_size,
                    video_files_size_in_mb,
                    src_duration,
                )
                # 后续继续追加到这个新目标文件时，偏移从该文件自身时长开始累计。
                current_offset = src_duration
            else:
                # 追加到已有目标 mp4 时，offset 必须是当前文件内累计时长，而不是所有文件总时长。
                videos_idx[key]["src_to_offset"][(src_chunk_idx, src_file_idx)] = current_offset
                videos_idx[key]["src_to_dst"][(src_chunk_idx, src_file_idx)] = (chunk_idx, file_idx)
                concatenate_video_files(
                    [dst_path, src_path],
                    dst_path,
                )
                logging.info(
                    "追加到目标视频: key=%s, src=(chunk-%03d,file-%03d), dst=(chunk-%03d,file-%03d), "
                    "offset=%.6f, src_duration=%.6f",
                    key,
                    src_chunk_idx,
                    src_file_idx,
                    chunk_idx,
                    file_idx,
                    current_offset,
                    src_duration,
                )
                current_offset += src_duration

            videos_idx[key]["episode_duration"] += src_duration

        videos_idx[key]["chunk"] = chunk_idx
        videos_idx[key]["file"] = file_idx
        videos_idx[key]["current_duration"] = current_offset
        logging.info(
            "视频流合并完成: key=%s, end_dst=(chunk-%03d,file-%03d), current_duration=%.6f, "
            "source_dataset_duration=%.6f",
            key,
            chunk_idx,
            file_idx,
            current_offset,
            videos_idx[key]["episode_duration"],
        )

    return videos_idx


def aggregate_data(src_meta, dst_meta, data_idx, data_files_size_in_mb, chunk_size):
    """Aggregates data chunks from a source dataset into the destination dataset.

    Reads source data files, updates indices to match the aggregated dataset,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        data_idx: Dictionary tracking data chunk and file indices.

    Returns:
        dict: Updated data_idx with current chunk and file indices.
    """
    unique_chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["data/chunk_index"], src_meta.episodes["data/file_index"], strict=False
        )
    }

    unique_chunk_file_ids = sorted(unique_chunk_file_ids)

    for src_chunk_idx, src_file_idx in unique_chunk_file_ids:
        src_path = src_meta.root / DEFAULT_DATA_PATH.format(
            chunk_index=src_chunk_idx, file_index=src_file_idx
        )
        df = pd.read_parquet(src_path)
        df = update_data_df(df, src_meta, dst_meta)

        data_idx = append_or_create_parquet_file(
            df,
            src_path,
            data_idx,
            data_files_size_in_mb,
            chunk_size,
            DEFAULT_DATA_PATH,
            contains_images=len(dst_meta.image_keys) > 0,
            aggr_root=dst_meta.root,
        )

    return data_idx


def aggregate_metadata(src_meta, dst_meta, meta_idx, data_idx, videos_idx):
    """Aggregates metadata from a source dataset into the destination dataset.

    Reads source metadata files, updates all indices and timestamps,
    and writes them to the destination with proper file rotation.

    Args:
        src_meta: Source dataset metadata.
        dst_meta: Destination dataset metadata.
        meta_idx: Dictionary tracking metadata chunk and file indices.
        data_idx: Dictionary tracking data chunk and file indices.
        videos_idx: Dictionary tracking video indices and timestamps.

    Returns:
        dict: Updated meta_idx with current chunk and file indices.
    """
    chunk_file_ids = {
        (c, f)
        for c, f in zip(
            src_meta.episodes["meta/episodes/chunk_index"],
            src_meta.episodes["meta/episodes/file_index"],
            strict=False,
        )
    }

    chunk_file_ids = sorted(chunk_file_ids)
    for chunk_idx, file_idx in chunk_file_ids:
        src_path = src_meta.root / DEFAULT_EPISODES_PATH.format(chunk_index=chunk_idx, file_index=file_idx)
        df = pd.read_parquet(src_path)
        df = update_meta_data(
            df,
            dst_meta,
            meta_idx,
            data_idx,
            videos_idx,
        )

        meta_idx = append_or_create_parquet_file(
            df,
            src_path,
            meta_idx,
            DEFAULT_DATA_FILE_SIZE_IN_MB,
            DEFAULT_CHUNK_SIZE,
            DEFAULT_EPISODES_PATH,
            contains_images=False,
            aggr_root=dst_meta.root,
        )

    # 保留全局累计时长，兼容没有 src_to_offset 的旧调用路径。
    # 正常聚合会使用 current_duration 按目标 mp4 文件内时间计算，因为 metadata 中的视频时间戳
    # 是相对于具体 mp4 文件的，而不是相对于所有视频文件的全局时间轴。
    for k in videos_idx:
        videos_idx[k]["latest_duration"] = videos_idx[k].get("latest_duration", 0) + videos_idx[k][
            "episode_duration"
        ]

    return meta_idx


def append_or_create_parquet_file(
    df: pd.DataFrame,
    src_path: Path,
    idx: dict[str, int],
    max_mb: float,
    chunk_size: int,
    default_path: str,
    contains_images: bool = False,
    aggr_root: Path = None,
):
    """Appends data to an existing parquet file or creates a new one based on size constraints.

    Manages file rotation when size limits are exceeded to prevent individual files
    from becoming too large. Handles both regular parquet files and those containing images.

    Args:
        df: DataFrame to write to the parquet file.
        src_path: Path to the source file (used for size estimation).
        idx: Dictionary containing current 'chunk' and 'file' indices.
        max_mb: Maximum allowed file size in MB before rotation.
        chunk_size: Maximum number of files per chunk before incrementing chunk index.
        default_path: Format string for generating file paths.
        contains_images: Whether the data contains images requiring special handling.
        aggr_root: Root path for the aggregated dataset.

    Returns:
        dict: Updated index dictionary with current chunk and file indices.
    """
    dst_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])

    if not dst_path.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        if contains_images:
            to_parquet_with_hf_images(df, dst_path)
        else:
            df.to_parquet(dst_path)
        return idx

    src_size = get_parquet_file_size_in_mb(src_path)
    dst_size = get_parquet_file_size_in_mb(dst_path)

    if dst_size + src_size >= max_mb:
        idx["chunk"], idx["file"] = update_chunk_file_indices(idx["chunk"], idx["file"], chunk_size)
        new_path = aggr_root / default_path.format(chunk_index=idx["chunk"], file_index=idx["file"])
        new_path.parent.mkdir(parents=True, exist_ok=True)
        final_df = df
        target_path = new_path
    else:
        existing_df = pd.read_parquet(dst_path)
        final_df = pd.concat([existing_df, df], ignore_index=True)
        target_path = dst_path

    if contains_images:
        to_parquet_with_hf_images(final_df, target_path)
    else:
        final_df.to_parquet(target_path)

    return idx


def finalize_aggregation(aggr_meta, all_metadata):
    """Finalizes the dataset aggregation by writing summary files and statistics.

    Writes the tasks file, info file with total counts and splits, and
    aggregated statistics from all source datasets.

    Args:
        aggr_meta: Aggregated dataset metadata.
        all_metadata: List of all source dataset metadata objects.
    """
    logging.info("write tasks")
    write_tasks(aggr_meta.tasks, aggr_meta.root)

    logging.info("write info")
    aggr_meta.info.update(
        {
            "total_tasks": len(aggr_meta.tasks),
            "total_episodes": sum(m.total_episodes for m in all_metadata),
            "total_frames": sum(m.total_frames for m in all_metadata),
            "splits": {"train": f"0:{sum(m.total_episodes for m in all_metadata)}"},
        }
    )
    write_info(aggr_meta.info, aggr_meta.root)

    logging.info("write stats")
    aggr_meta.stats = aggregate_stats([m.stats for m in all_metadata])
    write_stats(aggr_meta.stats, aggr_meta.root)
