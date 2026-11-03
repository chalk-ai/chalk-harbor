"""Post-train a model on a Harbor-based Chalk evaluation, with its scorers as the reward.

``train_policy`` is the training function a post-training workflow's training runs call
(``CHALK_TRAINING_MODULE=chalk_harbor.post_training.train_policy``): one on-policy GRPO step
of a LoRA adapter over the trajectories of one iteration's evaluation runs. The torch,
transformers and peft dependencies are the ``post-training`` extra, imported only when a
step runs.
"""

from chalk_harbor.post_training.config import ScorerColumn, TrainerConfig
from chalk_harbor.post_training.trainer import train_policy

__all__ = ["ScorerColumn", "TrainerConfig", "train_policy"]
