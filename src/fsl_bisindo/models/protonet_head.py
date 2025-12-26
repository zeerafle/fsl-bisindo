from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ProtoNetOutput:
    logits: torch.Tensor  # [n_query, n_way]
    prototypes: torch.Tensor  # [n_way, feature_dim]
    class_ids: torch.Tensor  # [n_way] (global labels)
    query_pred: torch.Tensor  # [n_query] predicted episodic labels (0..n_way-1)


class ProtoNetHead(nn.Module):
    """
    Prototypical Networks head operating on pre-extracted embeddings.
    Inputs:
        support: [n_support, feature_dim]
        support_labels: [n_support] global class labels (0..31)
        query: [n_query, feature_dim]

    Output logits are episodic: columns correspond to the selected classes
    in `class_ids` (sorted unique global labels).
    """

    def __init__(self, distance: str = "euclidean"):
        super().__init__()
        if distance not in {"euclidean"}:
            raise ValueError(f"Unsupported distance: {distance}")
        self.distance = distance

    def forward(
        self, support: torch.Tensor, support_labels: torch.Tensor, query: torch.Tensor
    ) -> ProtoNetOutput:
        if support.dim() != 2 or query.dim() != 2:
            raise ValueError("Support and query must be [N,D] tensors")

        support_labels = support_labels.long()
        class_ids = torch.unique(support_labels)
        class_ids, _ = torch.sort(class_ids)  # stable ordering

        # prototypes: [n_way, feature_dim]
        protos = []
        for c in class_ids:
            protos.append(support[support_labels == c].mean(dim=0, keepdim=True))
        prototypes = torch.cat(protos, dim=0)

        # logits: [n_query, n_way]
        if self.distance == "euclidean":
            dists = torch.cdist(query, prototypes)  # [n_query, n_way]
            logits = -dists  # negative distances as logits

        query_pred = torch.argmax(logits, dim=1)  # pyright: ignore[reportPossiblyUnboundVariable]

        return ProtoNetOutput(
            logits=logits,  # pyright: ignore[reportPossiblyUnboundVariable]
            prototypes=prototypes,
            class_ids=class_ids,
            query_pred=query_pred,
        )
