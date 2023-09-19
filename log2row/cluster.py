from __future__ import annotations

import os
import contextlib
import logging
from typing import Generator, List

import numpy
import torch

from . import Log, Match, Sink


@contextlib.contextmanager
def pool(model):
    pool = model.start_multi_process_pool()
    try:
        yield pool
    finally:
        model.stop_multi_process_pool(pool)


class Community:
    texts: List[str]
    embeddings: List[torch.Tensor]

    def __init__(self):
        self.texts = []
        self.embeddings = []

    def append(self, text: str, embedding: torch.Tensor):
        self.texts.append(text)
        self.embeddings.append(embedding)

    @classmethod
    def from_list(cls, texts: List[str], embeddings: List[torch.Tensor]):
        self = cls()
        self.texts = texts
        self.embeddings = embeddings
        return self

    def __iter__(self):
        for i in zip(self.texts, self.embeddings):
            yield i


def sample(community: torch.Tensor) -> List[int]:
    from sentence_transformers.util import cos_sim

    sims = cos_sim(community, community)

    min = (0, 1)
    for offset, sim in enumerate(sims):
        mean = sim.mean()
        if mean < min[1]:
            min = (offset, mean)

    sort = sorted(
        [(offset, sim) for offset, sim in enumerate(sims[min[0]].numpy())],
        key=lambda i: i[1],
    )

    samples = [community[min[0]], community[sort[0][0]]]

    sims = cos_sim(community, torch.from_numpy(numpy.array(samples)))

    min1 = (0, 1)
    for offset, sim in enumerate(sims):
        mean = sim.mean()
        if mean < min1[1]:
            min1 = (offset, mean)

    return [min[0], sort[0][0], min1[0]]


_path = os.path.dirname(__file__)


class Cluster:
    def __init__(
        self,
        sink: Sink,
        match: Match,
        model=os.path.join(_path, "../all-MiniLM-L6-v2"),
        buf_size=8 * 1024 * 1024,
        threshold=0.7,
        min_community_size=3,
    ):
        from sentence_transformers import SentenceTransformer as Embedder

        self.buf_size = buf_size
        self.threshold = threshold
        self.min_community_size = min_community_size
        self.sink = sink
        self.match = match
        self.model = Embedder(model)
        self.buffer = []
        self.size = 0

    def __iter__(self):
        import torch
        from .util import community_detection

        with pool(self.model) as p:
            for line in self.match:
                if isinstance(line, Log):
                    yield line
                    continue

                if line is not None:
                    self.buffer.append(line)
                    self.size += len(line)

                if self.size < self.buf_size and line is not None:
                    continue

                if self.size == 0:
                    continue

                logging.info(
                    f"embedding {format(self.size / 1024, '.2f')}KB / {len(self.buffer)} logs, it might take a while."
                )
                embeddings = torch.from_numpy(
                    self.model.encode_multi_process(self.buffer, p)
                )

                if len(embeddings) < 3:
                    continue

                logging.info("analyze log communities, it might take a while.")
                clusters = community_detection(
                    embeddings,
                    min_community_size=self.min_community_size,
                    threshold=self.threshold,
                )
                logging.info(f"get {len(clusters)} log communities.")

                if len(clusters) > 0:
                    yield from self._sample(clusters, embeddings)
                    self._recycle()
                else:
                    logging.warning(
                        f"no cluster is detected, maybe you should decrease the threshold."
                    )

    def _sample(self, clusters, embeddings):
        for cluster, _ in zip(clusters, range(0, 3)):
            vecs = [embeddings[i] for i in cluster]
            ids = sample(torch.from_numpy(numpy.array(vecs)))
            samples = Community.from_list(
                [self.buffer[cluster[id]] for id in ids],
                [embeddings[cluster[id]] for id in ids],
            )
            # sample_clusters.append(samples)

            logging.info(f"yield samples {samples.texts}")
            yield samples.texts

    def _recycle(self):
        send, self.buffer = self.buffer, []
        self.size = 0
        self.sink.send(send)
