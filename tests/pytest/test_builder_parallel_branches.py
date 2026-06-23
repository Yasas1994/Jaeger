from __future__ import annotations

import importlib.util

import pytest
import yaml

from jaeger.nnlib.builder import DynamicModelBuilder

HAS_TF = importlib.util.find_spec("tensorflow") is not None

pytestmark = pytest.mark.skipif(not HAS_TF, reason="tensorflow not installed")

MINIMAL_CONFIG = """
model:
  name: test_multiscale
  experiment: test
  seed: 42
  classifier_out_dim: 6
  base_dir: PLACEHOLDER
  class_label_map:
    - class: bacteria
      label: 0
    - class: phage
      label: 1
    - class: eukarya
      label: 2
    - class: archaea
      label: 3
    - class: plasmid
      label: 4
    - class: virus
      label: 5
  activation: gelu
  embedding:
    use_embedding_layer: true
    input_type: translated
    strands: 2
    frames: 6
    length: null
    input_shape: [6, null]
    embedding_size: 128
    embedding_regularizer: l2
    embedding_regularizer_w: 0.00001
  string_processor:
    data_format: numpy
    seq_onehot: false
    codon: CODON
    codon_id: CODON_ID
    crop_sizes: [500]
    validation_crop_sizes: [500]
    buffer_size: 5000
    shuffle: true
    reshuffle_each_iteration: true
    mutate: false
    mutation_rate: 0.05
    shuffle_frames: false
    masking: false
    classifier_labels: [0, 1, 2, 3, 4, 5]
    classifier_labels_map: [0, 1, 2, 3, 4, 5]
  representation_learner:
    hidden_layers:
      - name: masked_conv1d
        config:
          filters: 128
          kernel_size: 7
          strides: 1
          dilation_rate: 1
          use_bias: true
          activation: null
          kernel_regularizer: l2
          kernel_regularizer_w: 0.00001
      - name: masked_layernorm
      - name: activation
        config:
          activation: gelu
      - name: parallel_branches
        config:
          merge: concat
          branches:
            - hidden_layers:
                - name: residual_block
                  config:
                    use_1x1conv: false
                    block_size: 1
                    filters: 128
                    kernel_size: 5
                    dilation_rate: 6
                    use_bias: true
                    kernel_regularizer: l2
                    kernel_regularizer_w: 0.00001
                    norm_type: masked_layernorm
              pooling: max
            - hidden_layers:
                - name: residual_block
                  config:
                    use_1x1conv: false
                    block_size: 1
                    filters: 128
                    kernel_size: 5
                    dilation_rate: 14
                    use_bias: true
                    kernel_regularizer: l2
                    kernel_regularizer_w: 0.00001
                    norm_type: masked_layernorm
              pooling: max
    pooling: null
  classifier:
    input_shape: 256
    hidden_layers:
      - name: dropout
        config: { rate: 0.1 }
      - name: dense
        config:
          units: 6
          activation: null
          dtype: float32
          use_bias: true

training:
  data_dir: /tmp/jaeger_test_data
  experiment_root: experiments/experiment_{{ model.experiment }}_{{ model.seed }}
  classifier_dir: PLACEHOLDER
  classifier_epochs: 1
  classifier_train_steps: 2
  classifier_validation_steps: 1
  projection_epochs: 0
  reliability_epochs: 0
  batch_size: 64
  optimizer: adamw
  optimizer_params:
    learning_rate: 0.0003
    clipnorm: 1
  loss_classifier: categorical_crossentropy
  loss_params_classifier:
    from_logits: true
    label_smoothing: 0.1
  classifier_class_weights:
    0: 1.0
    1: 1.0
    2: 1.0
    3: 1.0
    4: 1.0
    5: 1.0
  metrics_classifier:
    - name: categorical_accuracy
      params: null
  callbacks:
    classifier: []
    projection: []
    reliability: []
  model_saving:
    path: '{{ model.base_dir }}/{{ training.experiment_root }}/model'
    save_weights: false
    save_exec_graph: false
  fragment_classifier_data:
    train: []
    validation: []
"""


def test_parallel_branches_builds_and_output_shape(tmp_path):
    cfg = yaml.safe_load(MINIMAL_CONFIG)
    base = tmp_path / "multiscale_test"
    cfg["model"]["base_dir"] = str(base)
    cfg["training"]["classifier_dir"] = str(base / "checkpoints" / "classifier")
    builder = DynamicModelBuilder(cfg)
    models = builder.build_fragment_classifier()
    rep_model = models["rep_model"]
    assert rep_model.output.shape[-1] == 256  # two 128-D branches concatenated
