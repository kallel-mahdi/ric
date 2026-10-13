"""NNX model state and reproducible random streams."""

from itertools import batched, islice

import jax
import numpy as np
from flax import nnx, struct
from flax.training import train_state

EVAL_SPLIT_IDS = {"val": 0, "test": 12}


class TrainState(train_state.TrainState):
    graphdef: nnx.GraphDef = struct.field(pytree_node=False)
    batch_stats: nnx.State

    def network(self, *, train: bool):
        model = nnx.merge(self.graphdef, self.params, self.batch_stats, copy=True)
        model.train() if train else model.eval()
        return model


@struct.dataclass
class LossOutput:
    losses: jax.Array
    probs: jax.Array
    metrics: dict[str, jax.Array]


def derive_seed_streams(seed: int):
    params_key, rollout_key, eval_key, shuffle_key = jax.random.split(jax.random.PRNGKey(seed), 4)
    hi, lo = np.asarray(shuffle_key, dtype=np.uint32)
    shuffle_seed = ((int(hi) << 32) | int(lo)) % (2**31 - 1)
    return params_key, rollout_key, eval_key, shuffle_seed


def batch_chunks(iterator, num_batches, chunk_size):
    for batches in batched(islice(iterator, num_batches), chunk_size):
        yield jax.tree.map(lambda *xs: np.stack(xs), *batches)
