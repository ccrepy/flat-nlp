# Copyright 2023 Flat NLP Authors.
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

"""Tests for pretrained_embedding_encoder."""

import numpy as np
import tensorflow as tf

from flat_nlp.encoding.lib import pretrained_embedding_encoder
from flat_nlp.encoding.lib import test_encoding_util

from absl.testing import absltest
from absl.testing import parameterized


@parameterized.parameters(
    (None),
    ('strip_any_punc'),
    ('strip_standalone_punc'),
)
class PretrainedEmbeddingEncoderTest(parameterized.TestCase):

  def test_encode(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    np.testing.assert_allclose(
        encoder.call(['a b', 'c']),
        [
            [  # "a b"
                [1., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 0., 0.],
            ],
            [  # "c"
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [3., 0., 0., 0.],
            ],
        ])

  def test_encode_cropped(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        2,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    self.assertEqual(encoder.n_vector, 2)
    np.testing.assert_allclose(
        encoder.call(['a b c']),
        [
            [  # "a b"
                [1., 0.],
                [0., 2.],
                [0., 0.],
            ],
        ])

  def test_encode_misadapted_token(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    encoder.adapt(['a b'])
    np.testing.assert_allclose(
        encoder.call(['a b c']),
        [
            [  # "a b c"
                [1., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 0., 0.],
            ],
        ])

  def test_encode_missing_token(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    with self.assertRaisesRegex(KeyError, "'d'"):
      encoder.call(['a d'])

  def test_encode_no_adaptation(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    with self.assertRaisesRegex(tf.errors.FailedPreconditionError,
                                'Table not initialized'):
      encoder.call(['a'])

  def test_encode_empty_input(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    with self.assertRaisesRegex(KeyError, 'pop from an empty set'):
      encoder.call([])

  def test_encode_empty_input_manual_adapt(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    encoder.adapt(['a b c'])

    with self.assertRaisesRegex(tf.errors.UnimplementedError,
                                'Cast float to string is not supported'):
      encoder.call([])

  def test_encode_empty_string_input(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    with self.assertRaisesRegex(
        ValueError, 'all the input array dimensions (except )?for the '
        'concatenation axis must match exactly'):
      encoder.call([''])

  def test_encode_empty_string_input_manual_adapt(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    encoder.adapt(['a b c'])
    np.testing.assert_allclose(
        encoder.call(['']), [
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
        ])

  def test_encode_blank_string_input(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    with self.assertRaisesRegex(
        ValueError, 'all the input array dimensions (except )?for the '
        'concatenation axis must match exactly'):
      encoder.call([''])

  def test_encode_blank_string_input_manual_adapt(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    encoder.adapt(['a b c'])
    np.testing.assert_allclose(
        encoder.call([' ']), [
            [
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
        ])

  def test_encode_with_null_token(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        null_token='d',
        normalization_mode=normalization_mode,
        oov_strategy=None)
    np.testing.assert_allclose(
        encoder.call(['a d c']),
        [
            [  # "a d c"
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 3., 0.],
            ],
        ])

  def test_encode_with_null_token_in_embedding_model(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        null_token='b',
        normalization_mode=normalization_mode,
        oov_strategy=None)
    np.testing.assert_allclose(
        encoder.call(['a b c']),
        [
            [  # "a b c"
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 3., 0.],
            ],
        ])

  def test_adapt_twice(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    encoder.adapt(['a'])

    try:
      encoder.adapt(['b c'])
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)

  def test_adapt_and_call(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    np.testing.assert_allclose(
        encoder.adapt_and_call(['a']),
        [
            [  # "a"
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
        ])

    np.testing.assert_allclose(
        encoder.adapt_and_call(['b c']),
        [
            [  # "b c"
                [0., 0., 0., 0.],
                [2., 0., 0., 0.],
                [0., 3., 0., 0.],
            ],
        ])

  def test_adapt_and_call_manual_adapt(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        adapt_before_call=False,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    np.testing.assert_allclose(
        encoder.adapt_and_call(['a']),
        [
            [  # "a"
                [1., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
            ],
        ])

    np.testing.assert_allclose(
        encoder.adapt_and_call(['b c']),
        [
            [  # "b c"
                [0., 0., 0., 0.],
                [2., 0., 0., 0.],
                [0., 3., 0., 0.],
            ],
        ])

  def test_invalid_max_sentence_length(self, normalization_mode):
    with self.assertRaisesRegex(ValueError,
                                'max_sentence_length must be a power of 2'):
      pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
          test_encoding_util.SIMPLE_EMBEDDING_MODEL,
          3,
          normalization_mode=normalization_mode,
          oov_strategy=None)

    try:
      pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
          test_encoding_util.SIMPLE_EMBEDDING_MODEL,
          4,
          normalization_mode=normalization_mode,
          oov_strategy=None)
    except Exception as e:  # pylint: disable=broad-except
      self.fail(e)


@parameterized.parameters(
    (None),
    ('strip_any_punc'),
    ('strip_standalone_punc'),
)
class PretrainedEmbeddingEncoderWithPunc(parameterized.TestCase):

  def test_adapt_with_punc(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    test_string = 'a b . b.'

    if normalization_mode is None:
      with self.assertRaisesRegex(KeyError, "('.')|('b.')"):
        encoder.adapt([test_string])

    if normalization_mode == 'strip_standalone_punc':
      with self.assertRaisesRegex(KeyError, "'b.'"):
        encoder.adapt([test_string])

    if normalization_mode == 'strip_any_punc':
      encoder.adapt([test_string])
      np.testing.assert_allclose(
          encoder.call([test_string]),
          [
              [  # "a b b"
                  [1., 0., 0., 0.],
                  [0., 2., 2., 0.],
                  [0., 0., 0., 0.],
              ],
          ])

  def test_encode_with_punc(self, normalization_mode):
    embedding_model = test_encoding_util.MockEmbeddingModel(
        3, {
            'a': (1, 0, 0),
            'b': (0, 2, 0),
            '.': (0, 0, 3),
            '<a>': (1, 0, 3),
        })

    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        embedding_model,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)

    if normalization_mode is None:
      np.testing.assert_allclose(
          encoder.call(['<a> b . b']),
          [
              [  # "<a> b . b"
                  [1., 0., 0., 0.],
                  [0., 2., 0., 2.],
                  [3., 0., 3., 0.],
              ],
          ])

    if normalization_mode == 'strip_standalone_punc':
      np.testing.assert_allclose(
          encoder.call(['<a> b . b']),
          [
              [  # "<a> b b"
                  [1., 0., 0., 0.],
                  [0., 2., 2., 0.],
                  [3., 0., 0., 0.],
              ],
          ])

    if normalization_mode == 'strip_any_punc':
      np.testing.assert_allclose(
          encoder.call(['<a> b . b']),
          [
              [  # "a b b"
                  [1., 0., 0., 0.],
                  [0., 2., 2., 0.],
                  [0., 0., 0., 0.],
              ],
          ])

  def test_encode(self, normalization_mode):
    encoder = pretrained_embedding_encoder.PretrainedEmbeddingEncoder(
        test_encoding_util.SIMPLE_EMBEDDING_MODEL,
        4,
        normalization_mode=normalization_mode,
        oov_strategy=None)
    np.testing.assert_allclose(
        encoder.call(['a b', 'c']),
        [
            [  # "a b"
                [1., 0., 0., 0.],
                [0., 2., 0., 0.],
                [0., 0., 0., 0.],
            ],
            [  # "c"
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [3., 0., 0., 0.],
            ],
        ])


if __name__ == '__main__':
  absltest.main()
