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

"""Tests for encoder_factory."""

from unittest import mock

from flat_nlp.encoding import encoder_factory
from flat_nlp.encoding.lib import test_encoding_util
from flat_nlp.public import flat_pb2
from absl.testing import absltest


class PretrainedEmbeddingEncoderIntegrationTest(absltest.TestCase):

  def test_no_normalization_mode(self):
    config = flat_pb2.EncoderConfig.PretrainedEmbeddingEncoderConfig(
        oov_strategy='')
    with self.assertRaisesRegex(ValueError,
                                'normalization_mode must be specified.'):
      _ = encoder_factory._load_pretrained_embedding_encoder(config)

  def test_no_oov_strategy(self):
    config = flat_pb2.EncoderConfig.PretrainedEmbeddingEncoderConfig(
        normalization_mode='')
    with self.assertRaisesRegex(ValueError, 'oov_strategy must be specified.'):
      _ = encoder_factory._load_pretrained_embedding_encoder(config)


@mock.patch(
    'flat_nlp.encoding.encoder_factory._load_base_encoder',
    lambda a: test_encoding_util.SIMPLE_TEXT8_ENCODER)
class EncoderAdapterIntegrationTest(absltest.TestCase):

  def test_base_encoder(self):
    encoder = encoder_factory.load_encoder(flat_pb2.EncoderConfig())
    self.assertFalse(encoder.is_output_hc)
    self.assertFalse(encoder.is_output_strided)

  def test_adapt_to_hc_output(self):
    encoder = encoder_factory.load_encoder(
        flat_pb2.EncoderConfig(adapt_to_hc_output=True))
    self.assertTrue(encoder.is_output_hc)
    self.assertFalse(encoder.is_output_strided)

  def test_adapt_to_strided_output(self):
    encoder = encoder_factory.load_encoder(
        flat_pb2.EncoderConfig(adapt_to_strided_output=True))
    self.assertFalse(encoder.is_output_hc)
    self.assertTrue(encoder.is_output_strided)

  def test_adapt_to_hc_strided_output(self):
    encoder = encoder_factory.load_encoder(
        flat_pb2.EncoderConfig(
            adapt_to_hc_output=True, adapt_to_strided_output=True))
    self.assertTrue(encoder.is_output_hc)
    self.assertTrue(encoder.is_output_strided)


class EncoderFactoryTest(absltest.TestCase):

  def test_load_encoder_tf_model_checkpoint(self):
    config = flat_pb2.EncoderConfig(tf_model_checkpoint='dummy_tf_model')

    with self.assertRaisesRegex(
        NotImplementedError, 'only pretrained_embedding_encoder is supported.'):
      _ = encoder_factory.load_encoder(config)

  def test_load_encoder_servomatic_model(self):
    config = flat_pb2.EncoderConfig(servomatic_model='dummy_servomatic_address')

    with self.assertRaisesRegex(
        NotImplementedError, 'only pretrained_embedding_encoder is supported.'):
      _ = encoder_factory.load_encoder(config)


if __name__ == '__main__':
  absltest.main()
