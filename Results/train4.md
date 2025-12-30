🚀 Starting training with parameters:
  - Model: microsoft/deberta-v3-base
2025-12-29 18:27:10.035164: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1767032830.054530     529 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1767032830.060415     529 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.
============================================================
🔍 DIMENSION DEBUG:
  Model: microsoft/deberta-v3-base
  emb_dim: 768
  hidden_dim: 384
  hidden_dim * 2 (bidirectional): 768
  deberta_feature_dim: 768
  gcn_dim: 768
  mem_dim: 768
============================================================
Parse dataset 'train': 100%|██████████████| 1264/1264 [00:00<00:00, 1572.53it/s]
Parse dataset 'test': 100%|█████████████████| 480/480 [00:00<00:00, 1730.75it/s]
    14res    8
config.json: 100%|██████████████████████████████| 579/579 [00:00<00:00, 641kB/s]
pytorch_model.bin: 100%|██████████████████████| 371M/371M [00:01<00:00, 280MB/s]
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing D2E2SModel: ['mask_predictions.classifier.weight', 'mask_predictions.LayerNorm.weight', 'mask_predictions.dense.bias', 'lm_predictions.lm_head.bias', 'mask_predictions.dense.weight', 'mask_predictions.classifier.bias', 'lm_predictions.lm_head.LayerNorm.bias', 'lm_predictions.lm_head.dense.weight', 'lm_predictions.lm_head.dense.bias', 'mask_predictions.LayerNorm.bias', 'deberta.embeddings.position_embeddings.weight', 'lm_predictions.lm_head.LayerNorm.weight']
- This IS expected if you are initializing D2E2SModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing D2E2SModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['entity_classifier.bias', 'fc.weight', 'attention_layer.linear_q.weight', 'TIN.GatedGCN.conv1.lin_l.bias', 'Sem_gcn.W.0.bias', 'TIN.lstm.bias_hh_l0', 'lstm.weight_ih_l0', 'attention_layer.w_query.weight', 'TIN.residual_layer3.3.weight', 'entity_classifier.weight', 'TIN.residual_layer4.3.weight', 'TIN.GatedGCN.conv3.att', 'TIN.lstm.weight_hh_l0_reverse', 'TIN.feature_fusion.2.bias', 'TIN.residual_layer3.0.weight', 'TIN.residual_layer4.3.bias', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer1.0.weight', 'lstm.weight_ih_l1', 'TIN.residual_layer4.0.bias', 'fc.bias', 'TIN.residual_layer3.2.bias', 'Syn_gcn.W.1.bias', 'lstm.bias_ih_l1', 'TIN.residual_layer3.0.bias', 'TIN.lstm.weight_hh_l0', 'lstm.weight_hh_l0', 'Sem_gcn.attn.linears.0.bias', 'TIN.GatedGCN.conv2.lin_r.weight', 'TIN.GatedGCN.conv1.lin_r.weight', 'TIN.GatedGCN.conv2.lin_r.bias', 'TIN.lstm.bias_hh_l1', 'TIN.GatedGCN.conv3.lin_l.bias', 'Sem_gcn.attn.linears.0.weight', 'lstm.bias_hh_l0_reverse', 'TIN.lstm.bias_ih_l0_reverse', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.feature_fusion.0.bias', 'TIN.residual_layer2.3.weight', 'size_embeddings.weight', 'TIN.lstm.bias_ih_l0', 'lstm.bias_ih_l0_reverse', 'lstm.weight_ih_l0_reverse', 'TIN.GatedGCN.conv1.lin_l.weight', 'TIN.lstm.weight_ih_l1', 'lstm.bias_hh_l1', 'Sem_gcn.attn.linears.1.bias', 'TIN.GatedGCN.conv1.lin_r.bias', 'Sem_gcn.W.1.weight', 'senti_classifier.weight', 'TIN.lstm.weight_hh_l1', 'attention_layer.w_value.bias', 'TIN.GatedGCN.conv3.lin_l.weight', 'lstm.bias_hh_l0', 'TIN.residual_layer1.0.bias', 'TIN.lstm.weight_hh_l1_reverse', 'attention_layer.linear_q.bias', 'Sem_gcn.W.1.bias', 'lstm.weight_hh_l1_reverse', 'lstm.weight_hh_l1', 'TIN.GatedGCN.conv2.lin_l.bias', 'attention_layer.w_query.bias', 'TIN.residual_layer2.3.bias', 'TIN.GatedGCN.conv3.lin_r.bias', 'TIN.GatedGCN.conv2.lin_l.weight', 'TIN.feature_fusion.3.bias', 'Sem_gcn.W.0.weight', 'TIN.residual_layer2.0.weight', 'TIN.residual_layer2.0.bias', 'Syn_gcn.W.0.bias', 'TIN.residual_layer4.0.weight', 'TIN.residual_layer4.2.weight', 'TIN.residual_layer1.2.bias', 'senti_classifier.bias', 'lstm.bias_ih_l0', 'TIN.lstm.bias_ih_l1_reverse', 'TIN.residual_layer1.3.bias', 'TIN.residual_layer1.3.weight', 'TIN.GatedGCN.conv3.lin_r.weight', 'TIN.residual_layer3.3.bias', 'lstm.bias_hh_l1_reverse', 'TIN.GatedGCN.conv2.bias', 'TIN.GatedGCN.conv1.att', 'lstm.weight_ih_l1_reverse', 'deberta.embeddings.position_ids', 'TIN.residual_layer4.2.bias', 'TIN.GatedGCN.conv1.bias', 'lstm.bias_ih_l1_reverse', 'TIN.residual_layer2.2.bias', 'TIN.feature_fusion.0.weight', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.feature_fusion.2.weight', 'lstm.weight_hh_l0_reverse', 'TIN.GatedGCN.conv2.att', 'TIN.GatedGCN.conv3.bias', 'TIN.feature_fusion.3.weight', 'TIN.residual_layer3.2.weight', 'Syn_gcn.W.0.weight', 'Syn_gcn.W.1.weight', 'TIN.lstm.bias_hh_l1_reverse', 'TIN.residual_layer1.2.weight', 'TIN.residual_layer2.2.weight', 'attention_layer.w_value.weight', 'TIN.lstm.bias_ih_l1', 'attention_layer.v.weight', 'TIN.lstm.weight_ih_l0_reverse', 'TIN.lstm.weight_ih_l0']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Train epoch 0: 100%|██████████████████████████| 158/158 [00:47<00:00,  3.34it/s]
Evaluate epoch 1: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 1: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 2: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 2: 100%|██████████████████████████| 158/158 [00:49<00:00,  3.22it/s]
Evaluate epoch 3: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        55.56         2.42         4.63          828
                   o         0.00         0.00         0.00          834

               micro        54.05         1.20         2.35         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        50.00         0.13         0.26          762

               micro        50.00         0.10         0.21          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 3: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.26it/s]
Evaluate epoch 4: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        56.88        21.98        31.71          828
                   o        83.33         1.20         2.36          834

               micro        57.83        11.55        19.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        30.30         1.31         2.52          762

               micro        30.30         1.03         1.99          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        12.12         0.52         1.01          762

               micro        12.12         0.41         0.80          971

Train epoch 4: 100%|██████████████████████████| 158/158 [00:49<00:00,  3.22it/s]
Evaluate epoch 5: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.33        22.71        35.21          828
                   o        71.43        13.19        22.27          834

               micro        75.63        17.93        28.99         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.72         7.48        12.97          762

               micro        48.72         5.87        10.48          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.72         7.48        12.97          762

               micro        48.72         5.87        10.48          971

Train epoch 5: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.26it/s]
Evaluate epoch 6: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        70.05        37.56        48.90          828
                   o        88.89         0.96         1.90          834

               micro        70.42        19.19        30.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        46.67         0.92         1.80          762

               micro        46.67         0.72         1.42          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        46.67         0.92         1.80          762

               micro        46.67         0.72         1.42          971

Train epoch 6: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 7: 100%|█████████████████████████| 60/60 [00:05<00:00, 10.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        59.57        53.74        56.51          828
                   o        77.09        25.42        38.23          834

               micro        64.29        39.53        48.96         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        40.55        21.39        28.01          762

               micro        40.45        16.79        23.73          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        40.55        21.39        28.01          762

               micro        40.45        16.79        23.73          971

Train epoch 7: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.24it/s]
Evaluate epoch 8: 100%|█████████████████████████| 60/60 [00:06<00:00,  9.79it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        45.09        67.03        53.91          828
                   o        80.50        27.22        40.68          834

               micro        51.69        47.05        49.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        19.05         2.70         4.73          148
                 NEU         0.00         0.00         0.00           61
                 POS        34.38        25.85        29.51          762

               micro        33.84        20.70        25.69          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         4.76         0.68         1.18          148
                 NEU         0.00         0.00         0.00           61
                 POS        34.21        25.72        29.36          762

               micro        33.16        20.29        25.18          971

Train epoch 8: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 9: 100%|█████████████████████████| 60/60 [00:06<00:00,  9.93it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        73.51        40.22        51.99          828
                   o        64.62        55.64        59.79          834

               micro        68.06        47.95        56.27         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        30.00         2.03         3.80          148
                 NEU         0.00         0.00         0.00           61
                 POS        46.42        34.91        39.85          762

               micro        46.14        27.70        34.62          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        30.00         2.03         3.80          148
                 NEU         0.00         0.00         0.00           61
                 POS        46.25        34.78        39.70          762

               micro        45.97        27.60        34.49          971

Train epoch 9: 100%|██████████████████████████| 158/158 [00:48<00:00,  3.24it/s]
Evaluate epoch 10: 100%|████████████████████████| 60/60 [00:06<00:00,  9.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        54.89        65.10        59.56          828
                   o        64.76        55.76        59.92          834

               micro        59.06        60.41        59.73         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        26.47         6.08         9.89          148
                 NEU         0.00         0.00         0.00           61
                 POS        32.53        45.93        38.08          762

               micro        32.34        36.97        34.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        26.47         6.08         9.89          148
                 NEU         0.00         0.00         0.00           61
                 POS        32.53        45.93        38.08          762

               micro        32.34        36.97        34.50          971

Train epoch 10: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 11: 100%|████████████████████████| 60/60 [00:06<00:00,  9.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        49.65        68.96        57.74          828
                   o        54.32        68.59        60.63          834

               micro        51.88        68.77        59.15         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        30.77         8.11        12.83          148
                 NEU         0.00         0.00         0.00           61
                 POS        27.42        52.49        36.02          762

               micro        27.50        42.43        33.37          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        30.77         8.11        12.83          148
                 NEU         0.00         0.00         0.00           61
                 POS        27.28        52.23        35.84          762

               micro        27.37        42.22        33.21          971

Train epoch 11: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 12: 100%|████████████████████████| 60/60 [00:06<00:00,  9.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        69.43        51.57        59.18          828
                   o        83.78        37.77        52.07          834

               micro        74.87        44.65        55.94         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        36.36         5.41         9.41          148
                 NEU         0.00         0.00         0.00           61
                 POS        54.27        30.05        38.68          762

               micro        53.38        24.41        33.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        36.36         5.41         9.41          148
                 NEU         0.00         0.00         0.00           61
                 POS        54.27        30.05        38.68          762

               micro        53.38        24.41        33.50          971

Train epoch 12: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.15it/s]
Evaluate epoch 13: 100%|████████████████████████| 60/60 [00:06<00:00,  9.77it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.05        38.65        51.70          828
                   o        76.51        54.68        63.78          834

               micro        77.14        46.69        58.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        31.58         8.11        12.90          148
                 NEU         0.00         0.00         0.00           61
                 POS        55.14        30.97        39.66          762

               micro        53.22        25.54        34.52          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        31.58         8.11        12.90          148
                 NEU         0.00         0.00         0.00           61
                 POS        55.14        30.97        39.66          762

               micro        53.22        25.54        34.52          971

Train epoch 13: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 14: 100%|████████████████████████| 60/60 [00:06<00:00,  9.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        67.82        51.93        58.82          828
                   o        79.91        42.93        55.85          834

               micro        72.83        47.41        57.43         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        41.67        10.14        16.30          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.85        30.71        37.71          762

               micro        48.35        25.64        33.51          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        41.67        10.14        16.30          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.85        30.71        37.71          762

               micro        48.35        25.64        33.51          971

Train epoch 14: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 15: 100%|████████████████████████| 60/60 [00:06<00:00,  9.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.18        44.57        56.77          828
                   o        71.89        61.03        66.02          834

               micro        74.41        52.83        61.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        38.78        12.84        19.29          148
                 NEU         0.00         0.00         0.00           61
                 POS        59.82        34.38        43.67          762

               micro        57.70        28.94        38.55          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        36.73        12.16        18.27          148
                 NEU         0.00         0.00         0.00           61
                 POS        59.82        34.38        43.67          762

               micro        57.49        28.84        38.41          971

Train epoch 15: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.13it/s]
Evaluate epoch 16: 100%|████████████████████████| 60/60 [00:06<00:00,  9.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        73.57        51.09        60.30          828
                   o        77.59        55.64        64.80          834

               micro        75.62        53.37        62.57         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        33.33        16.22        21.82          148
                 NEU         0.00         0.00         0.00           61
                 POS        57.23        37.93        45.62          762

               micro        54.15        32.23        40.41          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        33.33        16.22        21.82          148
                 NEU         0.00         0.00         0.00           61
                 POS        57.23        37.93        45.62          762

               micro        54.15        32.23        40.41          971

Train epoch 16: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.13it/s]
Evaluate epoch 17: 100%|████████████████████████| 60/60 [00:06<00:00,  8.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.88        44.20        56.66          828
                   o        70.55        63.19        66.67          834

               micro        73.74        53.73        62.16         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        35.00         9.46        14.89          148
                 NEU        50.00         1.64         3.17           61
                 POS        49.48        37.40        42.60          762

               micro        48.54        30.90        37.76          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        35.00         9.46        14.89          148
                 NEU        50.00         1.64         3.17           61
                 POS        49.48        37.40        42.60          762

               micro        48.54        30.90        37.76          971

Train epoch 17: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 18: 100%|████████████████████████| 60/60 [00:06<00:00,  9.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        84.65        39.98        54.31          828
                   o        81.28        53.12        64.25          834

               micro        82.69        46.57        59.58         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        33.33         8.11        13.04          148
                 NEU        33.33         1.64         3.12           61
                 POS        69.81        33.07        44.88          762

               micro        66.25        27.29        38.66          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        33.33         8.11        13.04          148
                 NEU        33.33         1.64         3.12           61
                 POS        69.81        33.07        44.88          762

               micro        66.25        27.29        38.66          971

Train epoch 18: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 19: 100%|████████████████████████| 60/60 [00:06<00:00,  9.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        66.76        58.21        62.19          828
                   o        73.22        62.95        67.70          834

               micro        69.98        60.59        64.95         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        35.71        10.14        15.79          148
                 NEU        50.00         1.64         3.17           61
                 POS        41.44        48.95        44.89          762

               micro        41.21        40.06        40.63          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        35.71        10.14        15.79          148
                 NEU        50.00         1.64         3.17           61
                 POS        41.44        48.95        44.89          762

               micro        41.21        40.06        40.63          971

Train epoch 19: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.10it/s]
Evaluate epoch 20: 100%|████████████████████████| 60/60 [00:06<00:00,  9.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        65.68        58.94        62.13          828
                   o        83.27        49.52        62.11          834

               micro        72.72        54.21        62.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        48.15         8.78        14.86          148
                 NEU        22.22         3.28         5.71           61
                 POS        52.91        40.55        45.91          762

               micro        52.26        33.37        40.73          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        48.15         8.78        14.86          148
                 NEU        22.22         3.28         5.71           61
                 POS        52.91        40.55        45.91          762

               micro        52.26        33.37        40.73          971

Train epoch 20: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 21: 100%|████████████████████████| 60/60 [00:06<00:00,  9.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        76.58        54.11        63.41          828
                   o        79.13        56.83        66.15          834

               micro        77.87        55.48        64.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        40.00         3.28         6.06           61
                 NEG        41.67        10.14        16.30          148
                 POS        60.94        40.94        48.98          762

               micro        59.49        33.88        43.18          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        40.00         3.28         6.06           61
                 NEG        41.67        10.14        16.30          148
                 POS        60.94        40.94        48.98          762

               micro        59.49        33.88        43.18          971

Train epoch 21: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.09it/s]
Evaluate epoch 22: 100%|████████████████████████| 60/60 [00:06<00:00,  9.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        71.70        59.06        64.77          828
                   o        80.27        56.12        66.06          834

               micro        75.65        57.58        65.39         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        45.16         9.46        15.64          148
                 POS        56.75        45.80        50.69          762

               micro        55.71        37.69        44.96          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        45.16         9.46        15.64          148
                 POS        56.75        45.80        50.69          762

               micro        55.71        37.69        44.96          971

Train epoch 22: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 23: 100%|████████████████████████| 60/60 [00:06<00:00,  9.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.94        56.04        64.49          828
                   o        79.03        58.75        67.40          834

               micro        77.50        57.40        65.95         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        37.14        17.57        23.85          148
                 NEU        28.57         3.28         5.88           61
                 POS        58.30        44.23        50.30          762

               micro        55.73        37.59        44.90          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        37.14        17.57        23.85          148
                 NEU        28.57         3.28         5.88           61
                 POS        58.30        44.23        50.30          762

               micro        55.73        37.59        44.90          971

Train epoch 23: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 24: 100%|████████████████████████| 60/60 [00:06<00:00,  9.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        70.08        61.96        65.77          828
                   o        79.45        58.87        67.63          834

               micro        74.37        60.41        66.67         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        16.67         1.64         2.99           61
                 NEG        44.83         8.78        14.69          148
                 POS        50.76        48.16        49.43          762

               micro        50.26        39.24        44.07          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        16.67         1.64         2.99           61
                 NEG        44.83         8.78        14.69          148
                 POS        50.76        48.16        49.43          762

               micro        50.26        39.24        44.07          971

Train epoch 24: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 25: 100%|████████████████████████| 60/60 [00:06<00:00,  8.87it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        63.40        65.70        64.53          828
                   o        72.84        65.59        69.02          834

               micro        67.81        65.64        66.71         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        25.00         6.56        10.39           61
                 NEG        42.86        16.22        23.53          148
                 POS        43.02        53.02        47.50          762

               micro        42.73        44.49        43.59          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        25.00         6.56        10.39           61
                 NEG        42.86        16.22        23.53          148
                 POS        43.02        53.02        47.50          762

               micro        42.73        44.49        43.59          971

Train epoch 25: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 26: 100%|████████████████████████| 60/60 [00:06<00:00,  9.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        84.58        47.71        61.00          828
                   o        83.13        57.31        67.85          834

               micro        83.78        52.53        64.57         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        47.22        11.49        18.48          148
                 NEU        12.50         1.64         2.90           61
                 POS        71.66        41.47        52.54          762

               micro        68.87        34.40        45.88          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        47.22        11.49        18.48          148
                 NEU        12.50         1.64         2.90           61
                 POS        71.66        41.47        52.54          762

               micro        68.87        34.40        45.88          971

Train epoch 26: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 27: 100%|████████████████████████| 60/60 [00:06<00:00,  8.96it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.34        59.42        66.44          828
                   o        77.81        63.07        69.67          834

               micro        76.60        61.25        68.07         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        33.75        18.24        23.68          148
                 POS        56.14        49.21        52.45          762

               micro        53.36        41.71        46.82          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        33.75        18.24        23.68          148
                 POS        56.14        49.21        52.45          762

               micro        53.36        41.71        46.82          971

Train epoch 27: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 28: 100%|████████████████████████| 60/60 [00:06<00:00,  9.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        82.70        52.54        64.25          828
                   o        79.97        59.35        68.13          834

               micro        81.22        55.96        66.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        40.00        16.22        23.08          148
                 NEU        23.08         4.92         8.11           61
                 POS        69.59        42.65        52.89          762

               micro        65.19        36.25        46.59          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        40.00        16.22        23.08          148
                 NEU        23.08         4.92         8.11           61
                 POS        69.59        42.65        52.89          762

               micro        65.19        36.25        46.59          971

Train epoch 28: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 29: 100%|████████████████████████| 60/60 [00:06<00:00,  9.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        67.15        66.91        67.03          828
                   o        78.01        62.11        69.16          834

               micro        71.99        64.50        68.04         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        20.00         3.28         5.63           61
                 NEG        36.99        18.24        24.43          148
                 POS        50.40        50.13        50.26          762

               micro        48.87        42.33        45.36          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        20.00         3.28         5.63           61
                 NEG        36.99        18.24        24.43          148
                 POS        50.40        50.13        50.26          762

               micro        48.87        42.33        45.36          971

Train epoch 29: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 30: 100%|████████████████████████| 60/60 [00:06<00:00,  9.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.30        57.37        66.57          828
                   o        76.79        63.07        69.26          834

               micro        77.96        60.23        67.96         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         7.69         1.64         2.70           61
                 NEG        41.94        17.57        24.76          148
                 POS        58.20        48.43        52.87          762

               micro        55.85        40.78        47.14          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         7.69         1.64         2.70           61
                 NEG        41.94        17.57        24.76          148
                 POS        58.20        48.43        52.87          762

               micro        55.85        40.78        47.14          971

Train epoch 30: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 31: 100%|████████████████████████| 60/60 [00:06<00:00,  9.06it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        82.32        53.99        65.21          828
                   o        81.99        57.31        67.47          834

               micro        82.15        55.66        66.36         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        33.72        19.59        24.79          148
                 NEU        33.33         4.92         8.57           61
                 POS        70.51        41.73        52.43          762

               micro        64.10        36.05        46.14          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        33.72        19.59        24.79          148
                 NEU        33.33         4.92         8.57           61
                 POS        70.51        41.73        52.43          762

               micro        64.10        36.05        46.14          971

Train epoch 31: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.08it/s]
Evaluate epoch 32: 100%|████████████████████████| 60/60 [00:06<00:00,  9.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.15        55.92        65.53          828
                   o        83.49        54.56        65.99          834

               micro        81.24        55.23        65.76         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        61.54        10.81        18.39          148
                 NEU        33.33         3.28         5.97           61
                 POS        64.48        43.83        52.19          762

               micro        64.00        36.25        46.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        61.54        10.81        18.39          148
                 NEU        33.33         3.28         5.97           61
                 POS        64.48        43.83        52.19          762

               micro        64.00        36.25        46.29          971

Train epoch 32: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 33: 100%|████████████████████████| 60/60 [00:06<00:00,  9.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        70.17        66.18        68.12          828
                   o        73.66        66.07        69.66          834

               micro        71.88        66.13        68.88         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        56.86        19.59        29.15          148
                 NEU        36.36         6.56        11.11           61
                 POS        45.24        53.67        49.10          762

               micro        45.76        45.52        45.64          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        56.86        19.59        29.15          148
                 NEU        36.36         6.56        11.11           61
                 POS        45.24        53.67        49.10          762

               micro        45.76        45.52        45.64          971

Train epoch 33: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.09it/s]
Evaluate epoch 34: 100%|████████████████████████| 60/60 [00:06<00:00,  8.94it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        73.60        63.65        68.26          828
                   o        78.23        64.63        70.78          834

               micro        75.87        64.14        69.51         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        50.85        20.27        28.99          148
                 NEU        40.00         3.28         6.06           61
                 POS        55.67        50.92        53.19          762

               micro        55.19        43.25        48.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        50.85        20.27        28.99          148
                 NEU        40.00         3.28         6.06           61
                 POS        55.67        50.92        53.19          762

               micro        55.19        43.25        48.50          971

Train epoch 34: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.09it/s]
Evaluate epoch 35: 100%|████████████████████████| 60/60 [00:06<00:00,  9.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        84.00        53.26        65.19          828
                   o        80.59        59.23        68.28          834

               micro        82.16        56.26        66.79         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        20.00         3.28         5.63           61
                 NEG        39.47        20.27        26.79          148
                 POS        75.82        42.39        54.38          762

               micro        69.34        36.56        47.88          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        20.00         3.28         5.63           61
                 NEG        39.47        20.27        26.79          148
                 POS        75.82        42.39        54.38          762

               micro        69.34        36.56        47.88          971

Train epoch 35: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.10it/s]
Evaluate epoch 36: 100%|████████████████████████| 60/60 [00:06<00:00,  9.33it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.03        56.64        66.34          828
                   o        80.53        61.99        70.05          834

               micro        80.29        59.33        68.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        49.02        16.89        25.13          148
                 NEU        13.64         4.92         7.23           61
                 POS        67.89        46.06        54.89          762

               micro        64.24        39.03        48.56          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        49.02        16.89        25.13          148
                 NEU        13.64         4.92         7.23           61
                 POS        67.89        46.06        54.89          762

               micro        64.24        39.03        48.56          971

Train epoch 36: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 37: 100%|████████████████████████| 60/60 [00:06<00:00,  9.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.97        59.42        67.44          828
                   o        76.67        64.63        70.14          834

               micro        77.29        62.03        68.83         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        32.23        26.35        29.00          148
                 NEU        25.00         3.28         5.80           61
                 POS        66.85        47.64        55.63          762

               micro        60.12        41.61        49.18          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        32.23        26.35        29.00          148
                 NEU        25.00         3.28         5.80           61
                 POS        66.85        47.64        55.63          762

               micro        60.12        41.61        49.18          971

Train epoch 37: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 38: 100%|████████████████████████| 60/60 [00:06<00:00,  9.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.74        61.71        69.19          828
                   o        75.67        67.87        71.55          834

               micro        77.09        64.80        70.42         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        14.29         3.28         5.33           61
                 NEG        35.79        22.97        27.98          148
                 POS        61.71        51.18        55.95          762

               micro        57.49        43.87        49.77          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        14.29         3.28         5.33           61
                 NEG        35.79        22.97        27.98          148
                 POS        61.71        51.18        55.95          762

               micro        57.49        43.87        49.77          971

Train epoch 38: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 39: 100%|████████████████████████| 60/60 [00:06<00:00,  9.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.94        53.86        64.68          828
                   o        81.89        61.27        70.10          834

               micro        81.45        57.58        67.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        25.00         3.28         5.80           61
                 NEG        37.84        18.92        25.23          148
                 POS        71.76        45.01        55.32          762

               micro        66.61        38.41        48.73          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        25.00         3.28         5.80           61
                 NEG        37.84        18.92        25.23          148
                 POS        71.76        45.01        55.32          762

               micro        66.61        38.41        48.73          971

Train epoch 39: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 40: 100%|████████████████████████| 60/60 [00:06<00:00,  9.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        82.71        53.74        65.15          828
                   o        80.98        61.27        69.76          834

               micro        81.78        57.52        67.54         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        38.96        20.27        26.67          148
                 NEU         0.00         0.00         0.00           61
                 POS        70.65        44.23        54.40          762

               micro        65.54        37.80        47.94          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        38.96        20.27        26.67          148
                 NEU         0.00         0.00         0.00           61
                 POS        70.65        44.23        54.40          762

               micro        65.54        37.80        47.94          971

Train epoch 40: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 41: 100%|████████████████████████| 60/60 [00:06<00:00,  9.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.90        58.57        67.60          828
                   o        84.09        59.59        69.75          834

               micro        81.97        59.09        68.67         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        25.00         4.92         8.22           61
                 NEG        54.17        17.57        26.53          148
                 POS        67.49        46.59        55.12          762

               micro        65.53        39.55        49.33          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        25.00         4.92         8.22           61
                 NEG        54.17        17.57        26.53          148
                 POS        67.49        46.59        55.12          762

               micro        65.53        39.55        49.33          971

Train epoch 41: 100%|█████████████████████████| 158/158 [00:52<00:00,  3.02it/s]
Evaluate epoch 42: 100%|████████████████████████| 60/60 [00:06<00:00,  9.29it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.00        61.47        68.37          828
                   o        81.59        62.71        70.92          834

               micro        79.26        62.09        69.64         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        42.31        22.30        29.20          148
                 POS        62.83        49.48        55.36          762

               micro        59.94        42.53        49.76          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        27.27         4.92         8.33           61
                 NEG        42.31        22.30        29.20          148
                 POS        62.83        49.48        55.36          762

               micro        59.94        42.53        49.76          971

Train epoch 42: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.13it/s]
Evaluate epoch 43: 100%|████████████████████████| 60/60 [00:06<00:00,  9.25it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.23        59.90        67.85          828
                   o        86.02        57.55        68.97          834

               micro        81.88        58.72        68.40         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        43.08        18.92        26.29          148
                 NEU        20.00         1.64         3.03           61
                 POS        70.22        45.80        55.44          762

               micro        66.67        38.93        49.15          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        43.08        18.92        26.29          148
                 NEU        20.00         1.64         3.03           61
                 POS        70.22        45.80        55.44          762

               micro        66.67        38.93        49.15          971

Train epoch 43: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 44: 100%|████████████████████████| 60/60 [00:06<00:00,  9.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        73.42        64.37        68.60          828
                   o        83.77        61.87        71.17          834

               micro        78.17        63.12        69.84         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        71.43        16.89        27.32          148
                 NEU        50.00         4.92         8.96           61
                 POS        60.34        50.52        55.00          762

               micro        60.82        42.53        50.06          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        71.43        16.89        27.32          148
                 NEU        50.00         4.92         8.96           61
                 POS        60.34        50.52        55.00          762

               micro        60.82        42.53        50.06          971

Train epoch 44: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.09it/s]
Evaluate epoch 45: 100%|████████████████████████| 60/60 [00:06<00:00,  8.98it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.28        64.37        69.40          828
                   o        79.51        66.07        72.17          834

               micro        77.37        65.22        70.78         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        44.44        21.62        29.09          148
                 NEU        50.00         4.92         8.96           61
                 POS        58.61        51.84        55.01          762

               micro        57.18        44.28        49.91          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        44.44        21.62        29.09          148
                 NEU        50.00         4.92         8.96           61
                 POS        58.61        51.84        55.01          762

               micro        57.18        44.28        49.91          971

Train epoch 45: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 46: 100%|████████████████████████| 60/60 [00:06<00:00,  9.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        81.24        58.57        68.07          828
                   o        85.49        57.91        69.05          834

               micro        83.30        58.24        68.56         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        59.62        20.95        31.00          148
                 NEU        27.27         4.92         8.33           61
                 POS        71.91        44.36        54.87          762

               micro        69.79        38.31        49.47          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        59.62        20.95        31.00          148
                 NEU        27.27         4.92         8.33           61
                 POS        71.91        44.36        54.87          762

               micro        69.79        38.31        49.47          971

Train epoch 46: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 47: 100%|████████████████████████| 60/60 [00:06<00:00,  9.25it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.19        61.71        68.59          828
                   o        80.98        65.35        72.33          834

               micro        79.10        63.54        70.47         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        47.17        16.89        24.88          148
                 NEU        16.67         1.64         2.99           61
                 POS        61.35        51.44        55.96          762

               micro        59.89        43.05        50.09          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        47.17        16.89        24.88          148
                 NEU        16.67         1.64         2.99           61
                 POS        61.35        51.44        55.96          762

               micro        59.89        43.05        50.09          971

Train epoch 47: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.10it/s]
Evaluate epoch 48: 100%|████████████████████████| 60/60 [00:06<00:00,  9.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.34        62.20        70.12          828
                   o        85.15        59.83        70.28          834

               micro        82.64        61.01        70.20         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        50.88        19.59        28.29          148
                 NEU        40.00         3.28         6.06           61
                 POS        71.23        47.77        57.19          762

               micro        68.94        40.68        51.17          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        50.88        19.59        28.29          148
                 NEU        40.00         3.28         6.06           61
                 POS        71.23        47.77        57.19          762

               micro        68.94        40.68        51.17          971

Train epoch 48: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 49: 100%|████████████████████████| 60/60 [00:06<00:00,  9.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.09        60.75        68.72          828
                   o        83.44        62.83        71.68          834

               micro        81.25        61.79        70.20         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        56.67        22.97        32.69          148
                 NEU        25.00         1.64         3.08           61
                 POS        67.27        48.82        56.58          762

               micro        65.96        41.92        51.26          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        56.67        22.97        32.69          148
                 NEU        25.00         1.64         3.08           61
                 POS        67.27        48.82        56.58          762

               micro        65.96        41.92        51.26          971

Train epoch 49: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 50: 100%|████████████████████████| 60/60 [00:06<00:00,  9.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.10        65.22        69.81          828
                   o        81.00        65.95        72.70          834

               micro        77.97        65.58        71.24         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        50.00        22.30        30.84          148
                 NEU        25.00         1.64         3.08           61
                 POS        60.90        51.71        55.93          762

               micro        59.69        44.08        50.71          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        50.00        22.30        30.84          148
                 NEU        25.00         1.64         3.08           61
                 POS        60.90        51.71        55.93          762

               micro        59.69        44.08        50.71          971

Train epoch 50: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 51: 100%|████████████████████████| 60/60 [00:06<00:00,  9.03it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.41        60.99        69.37          828
                   o        84.85        63.79        72.83          834

               micro        82.63        62.39        71.10         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        63.83        20.27        30.77          148
                 NEU        37.50         4.92         8.70           61
                 POS        69.23        49.61        57.80          762

               micro        68.39        42.33        52.29          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        63.83        20.27        30.77          148
                 NEU        37.50         4.92         8.70           61
                 POS        69.23        49.61        57.80          762

               micro        68.39        42.33        52.29          971

Train epoch 51: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.09it/s]
Evaluate epoch 52: 100%|████████████████████████| 60/60 [00:06<00:00,  9.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.48        62.32        69.08          828
                   o        85.36        61.51        71.50          834

               micro        81.22        61.91        70.26         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.86        16.22        25.95          148
                 NEU        14.29         1.64         2.94           61
                 POS        65.91        49.74        56.69          762

               micro        65.27        41.61        50.82          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.86        16.22        25.95          148
                 NEU        14.29         1.64         2.94           61
                 POS        65.91        49.74        56.69          762

               micro        65.27        41.61        50.82          971

Train epoch 52: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.13it/s]
Evaluate epoch 53: 100%|████████████████████████| 60/60 [00:06<00:00,  8.93it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.23        58.70        65.94          828
                   o        79.40        60.55        68.71          834

               micro        77.30        59.63        67.32         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        28.79        25.68        27.14          148
                 NEU        25.00         4.92         8.22           61
                 POS        67.74        44.09        53.42          762

               micro        58.91        38.83        46.80          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        28.79        25.68        27.14          148
                 NEU        25.00         4.92         8.22           61
                 POS        67.74        44.09        53.42          762

               micro        58.91        38.83        46.80          971

Train epoch 53: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.08it/s]
Evaluate epoch 54: 100%|████████████████████████| 60/60 [00:06<00:00,  9.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        76.14        64.73        69.97          828
                   o        81.11        59.71        68.78          834

               micro        78.45        62.21        69.40         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        42.50        22.97        29.82          148
                 NEU        27.27         4.92         8.33           61
                 POS        68.17        45.54        54.60          762

               micro        64.00        39.55        48.89          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        42.50        22.97        29.82          148
                 NEU        27.27         4.92         8.33           61
                 POS        68.17        45.54        54.60          762

               micro        64.00        39.55        48.89          971

Train epoch 54: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 55: 100%|████████████████████████| 60/60 [00:06<00:00,  9.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        70.14        61.84        65.73          828
                   o        83.80        53.36        65.20          834

               micro        75.89        57.58        65.48         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        45.45        13.51        20.83          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.98        44.23        46.48          762

               micro        48.70        36.77        41.90          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        45.45        13.51        20.83          148
                 NEU         0.00         0.00         0.00           61
                 POS        48.98        44.23        46.48          762

               micro        48.70        36.77        41.90          971

Train epoch 55: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 56: 100%|████████████████████████| 60/60 [00:06<00:00,  9.16it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        74.51        55.07        63.33          828
                   o        72.51        50.60        59.60          834

               micro        73.53        52.83        61.48         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        27.91        24.32        25.99          148
                 NEU        20.00         1.64         3.03           61
                 POS        67.96        34.51        45.78          762

               micro        57.58        30.90        40.21          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        27.91        24.32        25.99          148
                 NEU        20.00         1.64         3.03           61
                 POS        67.70        34.38        45.60          762

               micro        57.39        30.79        40.08          971

Train epoch 56: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.12it/s]
Evaluate epoch 57: 100%|████████████████████████| 60/60 [00:06<00:00,  9.08it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        72.20        53.02        61.14          828
                   o        72.86        60.19        65.92          834

               micro        72.55        56.62        63.60         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        37.70        15.54        22.01          148
                 NEU        25.00         3.28         5.80           61
                 POS        46.54        44.09        45.28          762

               micro        45.64        37.18        40.98          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        37.70        15.54        22.01          148
                 NEU        25.00         3.28         5.80           61
                 POS        46.54        44.09        45.28          762

               micro        45.64        37.18        40.98          971

Train epoch 57: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.15it/s]
Evaluate epoch 58: 100%|████████████████████████| 60/60 [00:06<00:00,  9.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        74.75        54.71        63.18          828
                   o        73.68        65.47        69.33          834

               micro        74.16        60.11        66.40         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        33.33         3.38         6.13          148
                 NEU        27.27         4.92         8.33           61
                 POS        53.20        50.13        51.62          762

               micro        52.42        40.16        45.48          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        33.33         3.38         6.13          148
                 NEU        27.27         4.92         8.33           61
                 POS        53.06        50.00        51.49          762

               micro        52.28        40.06        45.36          971

Train epoch 58: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.20it/s]
Evaluate epoch 59: 100%|████████████████████████| 60/60 [00:06<00:00,  9.97it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        74.73        58.94        65.90          828
                   o        81.79        62.47        70.84          834

               micro        78.22        60.71        68.36         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        66.67        10.81        18.60          148
                 NEU        20.00         1.64         3.03           61
                 POS        60.44        50.13        54.81          762

               micro        60.36        41.09        48.90          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        66.67        10.81        18.60          148
                 NEU        20.00         1.64         3.03           61
                 POS        60.44        50.13        54.81          762

               micro        60.36        41.09        48.90          971

Train epoch 59: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.23it/s]
Evaluate epoch 60: 100%|████████████████████████| 60/60 [00:05<00:00, 10.01it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        76.63        58.21        66.16          828
                   o        86.91        53.36        66.12          834

               micro        81.24        55.78        66.14         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        41.82        15.54        22.66          148
                 NEU       100.00         1.64         3.23           61
                 POS        75.46        43.18        54.92          762

               micro        71.75        36.35        48.26          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        41.82        15.54        22.66          148
                 NEU       100.00         1.64         3.23           61
                 POS        75.46        43.18        54.92          762

               micro        71.75        36.35        48.26          971

Train epoch 60: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 61: 100%|████████████████████████| 60/60 [00:06<00:00,  9.83it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        73.82        62.32        67.58          828
                   o        72.65        67.51        69.98          834

               micro        73.20        64.92        68.81         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU        28.57         3.28         5.88           61
                 NEG        29.93        29.73        29.83          148
                 POS        57.85        50.79        54.09          762

               micro        52.61        44.59        48.27          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU        28.57         3.28         5.88           61
                 NEG        29.93        29.73        29.83          148
                 POS        57.85        50.79        54.09          762

               micro        52.61        44.59        48.27          971

Train epoch 61: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 62: 100%|████████████████████████| 60/60 [00:06<00:00,  9.93it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        72.79        67.51        70.05          828
                   o        80.18        64.03        71.20          834

               micro        76.22        65.76        70.61         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        45.71        21.62        29.36          148
                 NEU        57.14         6.56        11.76           61
                 POS        60.38        53.81        56.90          762

               micro        58.99        45.93        51.65          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        45.71        21.62        29.36          148
                 NEU        57.14         6.56        11.76           61
                 POS        60.38        53.81        56.90          762

               micro        58.99        45.93        51.65          971

Train epoch 62: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.14it/s]
Evaluate epoch 63: 100%|████████████████████████| 60/60 [00:05<00:00, 10.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        74.83        65.34        69.76          828
                   o        82.82        64.75        72.68          834

               micro        78.62        65.04        71.19         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        45.83        22.30        30.00          148
                 NEU        40.00         6.56        11.27           61
                 POS        65.26        52.76        58.35          762

               micro        62.89        45.21        52.61          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        45.83        22.30        30.00          148
                 NEU        40.00         6.56        11.27           61
                 POS        65.26        52.76        58.35          762

               micro        62.89        45.21        52.61          971

Train epoch 63: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.26it/s]
Evaluate epoch 64: 100%|████████████████████████| 60/60 [00:05<00:00, 10.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.16        59.54        68.33          828
                   o        84.30        64.39        73.01          834

               micro        82.27        61.97        70.69         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        45.59        20.95        28.70          148
                 NEU        33.33         6.56        10.96           61
                 POS        71.37        49.74        58.62          762

               micro        67.76        42.64        52.34          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        45.59        20.95        28.70          148
                 NEU        33.33         6.56        10.96           61
                 POS        71.37        49.74        58.62          762

               micro        67.76        42.64        52.34          971

Train epoch 64: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 65: 100%|████████████████████████| 60/60 [00:05<00:00, 10.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.90        60.87        68.34          828
                   o        83.54        63.91        72.42          834

               micro        80.70        62.39        70.38         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        48.61        23.65        31.82          148
                 NEU        37.50         4.92         8.70           61
                 POS        70.29        48.43        57.34          762

               micro        67.27        41.92        51.65          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        48.61        23.65        31.82          148
                 NEU        37.50         4.92         8.70           61
                 POS        70.29        48.43        57.34          762

               micro        67.27        41.92        51.65          971

Train epoch 65: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 66: 100%|████████████████████████| 60/60 [00:06<00:00,  9.86it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.91        60.99        69.18          828
                   o        82.36        67.75        74.34          834

               micro        81.18        64.38        71.81         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        50.72        23.65        32.26          148
                 NEU        50.00         6.56        11.59           61
                 POS        69.46        52.23        59.63          762

               micro        67.23        45.01        53.92          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        50.72        23.65        32.26          148
                 NEU        50.00         6.56        11.59           61
                 POS        69.46        52.23        59.63          762

               micro        67.23        45.01        53.92          971

Train epoch 66: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 67: 100%|████████████████████████| 60/60 [00:05<00:00, 10.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        80.78        60.39        69.11          828
                   o        81.35        66.43        73.14          834

               micro        81.08        63.42        71.17         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        55.56        20.27        29.70          148
                 NEU        28.57         6.56        10.67           61
                 POS        68.97        51.05        58.67          762

               micro        66.93        43.56        52.78          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        55.56        20.27        29.70          148
                 NEU        28.57         6.56        10.67           61
                 POS        68.97        51.05        58.67          762

               micro        66.93        43.56        52.78          971

Train epoch 67: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 68: 100%|████████████████████████| 60/60 [00:05<00:00, 10.05it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.47        61.23        69.17          828
                   o        84.87        63.91        72.91          834

               micro        82.15        62.58        71.04         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        53.85        23.65        32.86          148
                 NEU        37.50         4.92         8.70           61
                 POS        71.94        50.13        59.09          762

               micro        69.54        43.25        53.33          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        53.85        23.65        32.86          148
                 NEU        37.50         4.92         8.70           61
                 POS        71.94        50.13        59.09          762

               micro        69.54        43.25        53.33          971

Train epoch 68: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 69: 100%|████████████████████████| 60/60 [00:05<00:00, 10.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        79.61        58.94        67.73          828
                   o        83.06        65.83        73.44          834

               micro        81.40        62.39        70.64         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.10        16.89        26.74          148
                 NEU        40.00         3.28         6.06           61
                 POS        69.20        51.31        58.93          762

               micro        68.64        43.05        52.91          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.10        16.89        26.74          148
                 NEU        40.00         3.28         6.06           61
                 POS        69.20        51.31        58.93          762

               micro        68.64        43.05        52.91          971

Train epoch 69: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.29it/s]
Evaluate epoch 70: 100%|████████████████████████| 60/60 [00:05<00:00, 10.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.54        62.56        69.25          828
                   o        84.68        64.27        73.07          834

               micro        81.01        63.42        71.14         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        58.06        24.32        34.29          148
                 NEU        37.50         4.92         8.70           61
                 POS        68.47        51.57        58.83          762

               micro        67.08        44.49        53.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        58.06        24.32        34.29          148
                 NEU        37.50         4.92         8.70           61
                 POS        68.47        51.57        58.83          762

               micro        67.08        44.49        53.50          971

Train epoch 70: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 71: 100%|████████████████████████| 60/60 [00:05<00:00, 10.01it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        77.75        64.13        70.28          828
                   o        85.99        64.75        73.87          834

               micro        81.69        64.44        72.05         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        61.82        22.97        33.50          148
                 NEU        42.86         4.92         8.82           61
                 POS        65.79        52.76        58.56          762

               micro        65.23        45.21        53.41          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        61.82        22.97        33.50          148
                 NEU        42.86         4.92         8.82           61
                 POS        65.79        52.76        58.56          762

               micro        65.23        45.21        53.41          971

Train epoch 71: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.26it/s]
Evaluate epoch 72: 100%|████████████████████████| 60/60 [00:06<00:00,  9.91it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.99        60.39        68.45          828
                   o        85.97        63.19        72.84          834

               micro        82.42        61.79        70.63         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        61.11        22.30        32.67          148
                 NEU        50.00         6.56        11.59           61
                 POS        71.21        50.00        58.75          762

               micro        70.02        43.05        53.32          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        61.11        22.30        32.67          148
                 NEU        50.00         6.56        11.59           61
                 POS        71.21        50.00        58.75          762

               micro        70.02        43.05        53.32          971

Train epoch 72: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 73: 100%|████████████████████████| 60/60 [00:05<00:00, 10.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.55        61.47        68.97          828
                   o        85.92        64.39        73.61          834

               micro        82.17        62.94        71.28         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        58.46        25.68        35.68          148
                 NEU        40.00         6.56        11.27           61
                 POS        72.62        50.13        59.32          762

               micro        70.55        43.67        53.94          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        58.46        25.68        35.68          148
                 NEU        40.00         6.56        11.27           61
                 POS        72.62        50.13        59.32          762

               micro        70.55        43.67        53.94          971

Train epoch 73: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 74: 100%|████████████████████████| 60/60 [00:05<00:00, 10.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.74        65.22        70.08          828
                   o        87.35        61.27        72.02          834

               micro        80.97        63.24        71.01         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        64.41        25.68        36.71          148
                 NEU        57.14         6.56        11.76           61
                 POS        67.65        51.05        58.19          762

               micro        67.24        44.39        53.47          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        64.41        25.68        36.71          148
                 NEU        57.14         6.56        11.76           61
                 POS        67.65        51.05        58.19          762

               micro        67.24        44.39        53.47          971

Train epoch 74: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 75: 100%|████████████████████████| 60/60 [00:05<00:00, 10.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        75.61        63.29        68.90          828
                   o        81.86        68.71        74.71          834

               micro        78.75        66.00        71.82         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        54.17        26.35        35.45          148
                 NEU        25.00         3.28         5.80           61
                 POS        62.27        53.28        57.43          762

               micro        61.07        46.04        52.50          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        54.17        26.35        35.45          148
                 NEU        25.00         3.28         5.80           61
                 POS        62.27        53.28        57.43          762

               micro        61.07        46.04        52.50          971

Train epoch 75: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 76: 100%|████████████████████████| 60/60 [00:05<00:00, 10.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        78.21        60.27        68.08          828
                   o        84.64        66.07        74.21          834

               micro        81.46        63.18        71.16         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        52.11        25.00        33.79          148
                 NEU        30.00         4.92         8.45           61
                 POS        68.05        51.71        58.76          762

               micro        65.76        44.70        53.22          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        52.11        25.00        33.79          148
                 NEU        30.00         4.92         8.45           61
                 POS        68.05        51.71        58.76          762

               micro        65.76        44.70        53.22          971

Train epoch 76: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 77: 100%|████████████████████████| 60/60 [00:05<00:00, 10.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        45.98         4.83         8.74          828
                   o        51.34        38.97        44.31          834

               micro        50.69        21.96        30.65         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        27.27         3.54         6.27          762

               micro        27.27         2.78         5.05          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS        27.27         3.54         6.27          762

               micro        27.27         2.78         5.05          971

Train epoch 77: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 78: 100%|████████████████████████| 60/60 [00:05<00:00, 10.97it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 78: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.28it/s]
Evaluate epoch 79: 100%|████████████████████████| 60/60 [00:05<00:00, 10.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 79: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 80: 100%|████████████████████████| 60/60 [00:05<00:00, 10.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 80: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.27it/s]
Evaluate epoch 81: 100%|████████████████████████| 60/60 [00:05<00:00, 10.65it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 81: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.18it/s]
Evaluate epoch 82: 100%|████████████████████████| 60/60 [00:06<00:00,  9.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 82: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.11it/s]
Evaluate epoch 83: 100%|████████████████████████| 60/60 [00:06<00:00,  9.57it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 83: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.05it/s]
Evaluate epoch 84: 100%|████████████████████████| 60/60 [00:06<00:00,  9.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 84: 100%|█████████████████████████| 158/158 [00:51<00:00,  3.07it/s]
Evaluate epoch 85: 100%|████████████████████████| 60/60 [00:06<00:00,  9.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 85: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.10it/s]
Evaluate epoch 86: 100%|████████████████████████| 60/60 [00:05<00:00, 10.19it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 86: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.20it/s]
Evaluate epoch 87: 100%|████████████████████████| 60/60 [00:05<00:00, 10.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 87: 100%|█████████████████████████| 158/158 [00:50<00:00,  3.16it/s]
Evaluate epoch 88: 100%|████████████████████████| 60/60 [00:07<00:00,  7.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 88: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.22it/s]
Evaluate epoch 89: 100%|████████████████████████| 60/60 [00:05<00:00, 10.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 89: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.21it/s]
Evaluate epoch 90: 100%|████████████████████████| 60/60 [00:05<00:00, 10.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 90: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.18it/s]
Evaluate epoch 91: 100%|████████████████████████| 60/60 [00:06<00:00,  9.78it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        42.11         0.97         1.89          828
                   o         0.00         0.00         0.00          834

               micro        42.11         0.48         0.95         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 91: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.20it/s]
Evaluate epoch 92: 100%|████████████████████████| 60/60 [00:05<00:00, 10.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        20.00         0.12         0.24          828
                   o         0.00         0.00         0.00          834

               micro        20.00         0.06         0.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 92: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.25it/s]
Evaluate epoch 93: 100%|████████████████████████| 60/60 [00:05<00:00, 10.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        33.33         0.12         0.24          828
                   o         0.00         0.00         0.00          834

               micro        33.33         0.06         0.12         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 93: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.23it/s]
Evaluate epoch 94: 100%|████████████████████████| 60/60 [00:05<00:00, 10.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        32.76         4.59         8.05          828
                   o         0.00         0.00         0.00          834

               micro        32.76         2.29         4.27         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 94: 100%|█████████████████████████| 158/158 [00:48<00:00,  3.24it/s]
Evaluate epoch 95: 100%|████████████████████████| 60/60 [00:05<00:00, 10.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        36.63         4.47         7.97          828
                   o         0.00         0.00         0.00          834

               micro        36.63         2.23         4.20         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 95: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.20it/s]
Evaluate epoch 96: 100%|████████████████████████| 60/60 [00:05<00:00, 10.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 96: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.21it/s]
Evaluate epoch 97: 100%|████████████████████████| 60/60 [00:06<00:00,  9.86it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 97: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.19it/s]
Evaluate epoch 98: 100%|████████████████████████| 60/60 [00:05<00:00, 10.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        32.82         7.73        12.51          828
                   o        18.42         0.84         1.61          834

               micro        30.47         4.27         7.49         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS         5.41         0.26         0.50          762

               micro         5.41         0.21         0.40          971


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          148
                 NEU         0.00         0.00         0.00           61
                 POS         5.41         0.26         0.50          762

               micro         5.41         0.21         0.40          971

Train epoch 98: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.22it/s]
Evaluate epoch 99: 100%|████████████████████████| 60/60 [00:05<00:00, 10.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t         0.00         0.00         0.00        828.0
                   o         0.00         0.00         0.00        834.0

               micro         0.00         0.00         0.00       1662.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0

Train epoch 99: 100%|█████████████████████████| 158/158 [00:49<00:00,  3.20it/s]
Evaluate epoch 100: 100%|███████████████████████| 60/60 [00:05<00:00, 10.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   t        46.67         1.69         3.26          828
                   o        23.53         0.48         0.94          834

               micro        38.30         1.08         2.11         1662


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        148.0
                 NEU         0.00         0.00         0.00         61.0
                 POS         0.00         0.00         0.00        762.0

               micro         0.00         0.00         0.00        971.0


Best F1 score: 53.9440203562341 at epoch 73