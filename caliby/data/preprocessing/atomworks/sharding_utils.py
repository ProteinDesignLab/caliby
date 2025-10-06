import math

from natsort import natsorted


def take_shard(items: list, *, shard_id: int | None, num_shards: int, keep_order: bool = False) -> list:
    """
    Return the subset of `items` assigned to this shard. If shard_id is None, do not shard.
    - shard_id: the id of the shard.
    - num_shards: the total number of shards
    - keep_order: keep the original order of the items. By default, the items are natsorted to ensure consistent shard ordering.
    """
    if not use_sharding(shard_id, num_shards):
        return items

    if not keep_order:
        items = natsorted(items)

    chunk_size = math.ceil(len(items) / num_shards)
    start_idx = shard_id * chunk_size
    end_idx = min(start_idx + chunk_size, len(items))
    items = items[start_idx:end_idx]
    return items


def use_sharding(shard_id: int | None, num_shards: int) -> bool:
    """
    Determine if sharding should be used.
    """
    return (shard_id is not None) and (num_shards > 1)
