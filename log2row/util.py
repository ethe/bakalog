import contextlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Tuple


@dataclass
class Log:
    pattern: str
    content: str
    groups: Tuple[str, ...]


class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class Singleton(metaclass=SingletonMeta):
    pass


class Serializer(json.JSONEncoder, metaclass=SingletonMeta):
    def __init__(self, *args, **kwargs):
        self.serializers = {}
        super().__init__(*args, **kwargs)

    def default(self, o: Any) -> Any:
        if type(o) in self.serializers:
            return self.serializers[type(o)](o)
        return super().default(o)

    def register(self, type, encoder):
        self.serializers[type] = encoder


class Memory(Singleton):
    home = os.environ["HOME"]
    PATH = f"{home}/.log2row"

    def __init__(
        self,
    ):
        if not os.path.exists(self.PATH):
            os.mkdir(self.PATH)

    @contextlib.contextmanager
    def current(self, file: str):
        with open(f"{self.PATH}/patterns", "a+") as f:
            f.seek(0)
            if f.read() == "":
                f.write(r"{}")
                f.seek(0)
            f.seek(0)
            all = json.loads(f.read())
            if file not in all:
                all[file] = {}
            self.memory = all[file]
            try:
                yield
            finally:
                f.seek(0)
                f.truncate()
                f.write(json.dumps(all, cls=Serializer))

    @classmethod
    def serialize(cls, type, serializer):
        Serializer().register(type, serializer)

    def load(self, field, init):
        if field not in self.memory:
            self.memory[field] = init
        return self.memory[field]


def community_detection(
    embeddings, threshold=0.75, min_community_size=3, batch_size=1024
):
    """
    Function for Fast Community Detection
    Finds in the embeddings all communities, i.e. embeddings that are close (closer than threshold).
    Returns only communities that are larger than min_community_size. The communities are returned
    in decreasing order. The first element in each list is the central point in the community.
    """

    import torch
    from sentence_transformers.util import cos_sim

    threshold = torch.tensor(threshold, device=embeddings.device)

    extracted_communities = []

    # Maximum size for community
    min_community_size = min(min_community_size, len(embeddings))
    sort_max_size = min(max(2 * min_community_size, 50), len(embeddings))

    for start_idx in range(0, len(embeddings), batch_size):
        # Compute cosine similarity scores
        cos_scores = cos_sim(embeddings[start_idx : start_idx + batch_size], embeddings)

        # Minimum size for a community
        top_k_values, _ = cos_scores.topk(k=min_community_size, largest=True)

        # Filter for rows >= min_threshold
        for i in range(len(top_k_values)):
            if top_k_values[i][-1] >= threshold:
                new_cluster = []

                # Only check top k most similar entries
                top_val_large, top_idx_large = cos_scores[i].topk(
                    k=sort_max_size, largest=True
                )

                # Check if we need to increase sort_max_size
                while top_val_large[-1] > threshold:
                    sort_max_size = min(2 * sort_max_size, len(embeddings))
                    top_val_large, top_idx_large = cos_scores[i].topk(
                        k=sort_max_size, largest=True
                    )
                    if min(2 * sort_max_size, len(embeddings)) <= sort_max_size:
                        break

                for idx, val in zip(top_idx_large.tolist(), top_val_large):
                    if val < threshold:
                        break

                    new_cluster.append(idx)

                extracted_communities.append(new_cluster)

        del cos_scores

    # Largest cluster first
    extracted_communities = sorted(
        extracted_communities, key=lambda x: len(x), reverse=True
    )

    # Step 2) Remove overlapping communities
    unique_communities = []
    extracted_ids = set()

    for _, community in enumerate(extracted_communities):
        community = sorted(community)
        non_overlapped_community = []
        for idx in community:
            if idx not in extracted_ids:
                non_overlapped_community.append(idx)

        if len(non_overlapped_community) >= min_community_size:
            unique_communities.append(non_overlapped_community)
            extracted_ids.update(non_overlapped_community)

    unique_communities = sorted(unique_communities, key=lambda x: len(x), reverse=True)

    return unique_communities


units = {"B": 1, "KB": 2**10, "MB": 2**20, "GB": 2**30, "TB": 2**40}


def parse_size(size):
    size = size.upper()
    if not re.match(r" ", size):
        size = re.sub(r"([KMGT]?B)", r" \1", size)
    number, unit = [string.strip() for string in size.split()]
    return int(float(number) * units[unit])
