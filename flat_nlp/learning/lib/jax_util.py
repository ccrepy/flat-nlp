# Copyright 2022 Flat NLP Authors.
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

"""Utility library for learning with Jax."""

import flax.linen as nn
import jax
import jax.numpy as jnp
import tensorflow as tf

from flat_nlp.encoding.lib import embedding_util


@jax.jit
def l2_normalize(x: jnp.ndarray, epsilon: float = 1e-9) -> jnp.ndarray:
  """Applies L2 normalization."""
  l2_norm = jnp.sum(jax.lax.square(x), axis=-1, keepdims=True)
  return jnp.asarray(x * jax.lax.rsqrt(l2_norm + epsilon))


def vmap_product(fn, x, y):
  """Applies a function to the carthesian product of each pairs."""
  fn_ = jax.vmap(
      jax.vmap(
          lambda _x, _y: fn(jnp.array([_x]), jnp.array([_y])),
          in_axes=(0, None),
      ),
      in_axes=(None, 0),
  )
  return jnp.squeeze(fn_(x, y), axis=-1)


def build_pretrained_embeddings_module(
    adapted_encoder: tf.keras.layers.TextVectorization,
    pretrained_embeddings_model: embedding_util.EmbeddingModel,
    normalize_embeddings: bool = False,
    immutable_embeddings: bool = True,):
  """Builds a Jax module to expose some pretrained embeddings model."""

  all_vocabulary = adapted_encoder.get_vocabulary()
  encoder_tokens, vocabulary = all_vocabulary[:2], all_vocabulary[2:]

  vocabulary_embeddings = pretrained_embeddings_model[vocabulary]

  all_embeddings = jnp.concatenate([
      jnp.zeros((len(encoder_tokens), pretrained_embeddings_model.vector_size)),
      vocabulary_embeddings,
  ])

  class JaxPretrainedEmbedding(nn.Module):
    """Module to encode text as nd signal based on pretrained embeddings values."""

    normalize_embeddings: bool
    immutable_embeddings: bool

    def setup(self):
      self.embeddings = nn.Embed(len(all_vocabulary),
                                 pretrained_embeddings_model.vector_size,
                                 embedding_init=lambda *args: all_embeddings)

    def lookup_and_encode(self, x):
      """Lookups and encodes tokens into nd signals."""
      embeddings = self.lookup(x)
      if self.immutable_embeddings:
        embeddings = jax.lax.stop_gradient(embeddings)
      return embeddings.swapaxes(2, 1)

    def lookup(self, x):
      """Lookups l2 normalized vector for tokens."""

      mask = jnp.repeat(
          (x <= 1)[..., jnp.newaxis],
          pretrained_embeddings_model.vector_size,
          axis=-1,
      )

      if self.normalize_embeddings:
        l2_emneddings = l2_normalize(self.embeddings(x))
        return jnp.where(mask, 0., l2_emneddings)
      else:
        raw_embeddings = self.embeddings(x)
        return jnp.where(mask, 0., raw_embeddings)

  return JaxPretrainedEmbedding(normalize_embeddings, immutable_embeddings)
