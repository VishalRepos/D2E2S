🚀 Starting training with parameters:
  - Model: microsoft/deberta-v3-base
  - emb_dim: 768
  - hidden_dim: 384 (bidirectional → 768)
  - deberta_feature_dim: 768
  - gcn_dim: 768
  - mem_dim: 768
============================================================
2025-11-26 06:10:15.561910: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1764137415.762377     174 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1764137415.821384     174 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
tokenizer_config.json: 100%|██████████████████| 52.0/52.0 [00:00<00:00, 398kB/s]
config.json: 100%|█████████████████████████████| 633/633 [00:00<00:00, 2.98MB/s]
spm.model: 100%|███████████████████████████| 2.45M/2.45M [00:00<00:00, 5.77MB/s]
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
Parse dataset 'train': 100%|████████████████| 592/592 [00:00<00:00, 1758.36it/s]
Parse dataset 'test': 100%|██████████████████| 320/320 [00:00<00:00, 364.98it/s]
    15res    8
config.json: 100%|█████████████████████████████| 579/579 [00:00<00:00, 3.91MB/s]
pytorch_model.bin: 100%|██████████████████████| 371M/371M [00:01<00:00, 226MB/s]
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing D2E2SModel: ['mask_predictions.classifier.bias', 'mask_predictions.dense.weight', 'lm_predictions.lm_head.dense.bias', 'lm_predictions.lm_head.LayerNorm.bias', 'lm_predictions.lm_head.bias', 'lm_predictions.lm_head.LayerNorm.weight', 'mask_predictions.dense.bias', 'mask_predictions.LayerNorm.weight', 'mask_predictions.classifier.weight', 'deberta.embeddings.position_embeddings.weight', 'lm_predictions.lm_head.dense.weight', 'mask_predictions.LayerNorm.bias']
- This IS expected if you are initializing D2E2SModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing D2E2SModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['TIN.GatedGCN.conv2.lin_r.weight', 'TIN.residual_layer1.3.bias', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer4.3.bias', 'TIN.residual_layer3.0.bias', 'TIN.residual_layer3.2.bias', 'TIN.feature_fusion.2.weight', 'TIN.residual_layer1.3.weight', 'TIN.GatedGCN.conv3.lin_l.weight', 'TIN.GatedGCN.conv1.bias', 'TIN.residual_layer2.0.bias', 'TIN.residual_layer2.0.weight', 'TIN.lstm.weight_hh_l1', 'lstm.bias_ih_l1', 'TIN.GatedGCN.conv3.lin_r.bias', 'Syn_gcn.W.0.bias', 'lstm.bias_ih_l0_reverse', 'TIN.GatedGCN.conv3.att', 'lstm.bias_ih_l0', 'TIN.GatedGCN.conv2.att', 'TIN.residual_layer1.2.bias', 'fc.bias', 'TIN.GatedGCN.conv1.lin_r.weight', 'TIN.lstm.bias_hh_l0', 'lstm.bias_hh_l0', 'TIN.residual_layer4.0.weight', 'TIN.GatedGCN.conv1.lin_l.weight', 'TIN.residual_layer2.3.bias', 'TIN.lstm.weight_ih_l0', 'Syn_gcn.W.1.weight', 'lstm.weight_hh_l0', 'TIN.GatedGCN.conv1.lin_l.bias', 'Sem_gcn.attn.linears.0.weight', 'TIN.lstm.bias_hh_l1', 'TIN.residual_layer3.2.weight', 'attention_layer.w_value.bias', 'TIN.GatedGCN.conv3.lin_l.bias', 'Sem_gcn.W.0.weight', 'TIN.GatedGCN.conv2.lin_r.bias', 'attention_layer.w_value.weight', 'TIN.lstm.bias_ih_l0', 'attention_layer.linear_q.weight', 'Syn_gcn.W.1.bias', 'TIN.residual_layer2.2.bias', 'Sem_gcn.W.1.weight', 'senti_classifier.weight', 'TIN.residual_layer4.0.bias', 'TIN.feature_fusion.0.bias', 'TIN.residual_layer1.0.bias', 'TIN.feature_fusion.3.bias', 'TIN.lstm.weight_hh_l1_reverse', 'attention_layer.w_query.weight', 'Sem_gcn.W.0.bias', 'TIN.residual_layer3.0.weight', 'lstm.weight_hh_l0_reverse', 'TIN.lstm.weight_hh_l0', 'TIN.feature_fusion.3.weight', 'TIN.residual_layer4.2.weight', 'lstm.bias_ih_l1_reverse', 'TIN.residual_layer4.2.bias', 'TIN.residual_layer4.3.weight', 'TIN.lstm.bias_ih_l1_reverse', 'entity_classifier.weight', 'TIN.residual_layer1.0.weight', 'fc.weight', 'lstm.weight_ih_l0', 'TIN.residual_layer3.3.bias', 'TIN.GatedGCN.conv2.lin_l.weight', 'entity_classifier.bias', 'lstm.weight_hh_l1', 'lstm.weight_ih_l0_reverse', 'TIN.GatedGCN.conv2.lin_l.bias', 'TIN.GatedGCN.conv1.lin_r.bias', 'Sem_gcn.attn.linears.0.bias', 'TIN.feature_fusion.2.bias', 'attention_layer.v.weight', 'TIN.lstm.weight_ih_l1', 'attention_layer.linear_q.bias', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.GatedGCN.conv2.bias', 'size_embeddings.weight', 'TIN.lstm.bias_hh_l1_reverse', 'lstm.weight_ih_l1_reverse', 'TIN.lstm.weight_hh_l0_reverse', 'TIN.residual_layer2.3.weight', 'deberta.embeddings.position_ids', 'lstm.weight_ih_l1', 'Sem_gcn.attn.linears.1.bias', 'Sem_gcn.W.1.bias', 'lstm.bias_hh_l1', 'TIN.GatedGCN.conv1.att', 'TIN.lstm.weight_ih_l0_reverse', 'TIN.GatedGCN.conv3.bias', 'lstm.bias_hh_l1_reverse', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.lstm.bias_ih_l0_reverse', 'TIN.GatedGCN.conv3.lin_r.weight', 'TIN.residual_layer2.2.weight', 'TIN.residual_layer3.3.weight', 'Syn_gcn.W.0.weight', 'TIN.residual_layer1.2.weight', 'senti_classifier.bias', 'attention_layer.w_query.bias', 'lstm.bias_hh_l0_reverse', 'TIN.lstm.bias_ih_l1', 'lstm.weight_hh_l1_reverse', 'TIN.feature_fusion.0.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Train epoch 0: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 1: 100%|█████████████████████████| 40/40 [00:05<00:00,  7.99it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 1: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 2: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.25it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 2: 100%|████████████████████████████| 74/74 [00:22<00:00,  3.23it/s]
Evaluate epoch 3: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 3: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 4: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        53.33        13.02        20.93          430

               micro        53.33         6.30        11.27          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        15.79         0.95         1.79          316
                 NEU         0.00         0.00         0.00           25

               micro        15.79         0.62         1.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 4: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 5: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        68.75         4.79         8.96          459
                   t        76.34        23.26        35.65          430

               micro        74.85        13.72        23.19          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        42.31         3.48         6.43          316
                 NEU         0.00         0.00         0.00           25

               micro        39.29         2.28         4.31          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        42.31         3.48         6.43          316
                 NEU         0.00         0.00         0.00           25

               micro        39.29         2.28         4.31          483

Train epoch 5: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 6: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.95        40.74        44.05          459
                   t        75.82        32.09        45.10          430

               micro        56.82        36.56        44.49          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        31.98        22.47        26.39          316
                 NEU         0.00         0.00         0.00           25

               micro        31.84        14.70        20.11          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        30.63        21.52        25.28          316
                 NEU         0.00         0.00         0.00           25

               micro        30.49        14.08        19.26          483

Train epoch 6: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 7: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.32it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        62.20        34.42        44.32          459
                   t        82.84        32.56        46.74          430

               micro        70.45        33.52        45.43          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        39.43        21.84        28.11          316
                 NEU         0.00         0.00         0.00           25

               micro        39.20        14.29        20.94          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        39.43        21.84        28.11          316
                 NEU         0.00         0.00         0.00           25

               micro        39.20        14.29        20.94          483

Train epoch 7: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 8: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         0.44         0.87          459
                   t         0.00         0.00         0.00          430

               micro       100.00         0.22         0.45          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 8: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 9: 100%|█████████████████████████| 40/40 [00:04<00:00,  8.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         1.31         2.58          459
                   t         0.00         0.00         0.00          430

               micro       100.00         0.67         1.34          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 9: 100%|████████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 10: 100%|████████████████████████| 40/40 [00:04<00:00,  8.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        83.33         2.18         4.25          459
                   t        11.11         0.23         0.46          430

               micro        52.38         1.24         2.42          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 10: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 11: 100%|████████████████████████| 40/40 [00:04<00:00,  8.32it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.11         5.01         9.13          459
                   t         0.00         0.00         0.00          430

               micro        50.00         2.59         4.92          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        14.29         0.95         1.78          316
                 NEU         0.00         0.00         0.00           25

               micro        14.29         0.62         1.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 11: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 12: 100%|████████████████████████| 40/40 [00:04<00:00,  8.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         0.87         1.73          459
                   t         0.00         0.00         0.00          430

               micro        80.00         0.45         0.89          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 12: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 13: 100%|████████████████████████| 40/40 [00:04<00:00,  8.19it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        85.71         2.61         5.07          459
                   t        25.00         0.23         0.46          430

               micro        72.22         1.46         2.87          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 13: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 14: 100%|████████████████████████| 40/40 [00:04<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.02         5.45         9.84          459
                   t        25.00         0.47         0.91          430

               micro        47.37         3.04         5.71          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        22.22         0.63         1.23          316
                 NEU         0.00         0.00         0.00           25

               micro        22.22         0.41         0.81          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        11.11         0.32         0.62          316
                 NEU         0.00         0.00         0.00           25

               micro        11.11         0.21         0.41          483

Train epoch 14: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 15: 100%|████████████████████████| 40/40 [00:04<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.57         2.40         4.65          459
                   t        40.00         0.47         0.92          430

               micro        68.42         1.46         2.86          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        40.00         0.63         1.25          316
                 NEU         0.00         0.00         0.00           25

               micro        40.00         0.41         0.82          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        40.00         0.63         1.25          316
                 NEU         0.00         0.00         0.00           25

               micro        40.00         0.41         0.82          483

Train epoch 15: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 16: 100%|████████████████████████| 40/40 [00:04<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.83         2.40         4.55          459
                   t         0.00         0.00         0.00          430

               micro        37.93         1.24         2.40          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 16: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.24it/s]
Evaluate epoch 17: 100%|████████████████████████| 40/40 [00:04<00:00,  8.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        62.50         3.27         6.21          459
                   t        30.43         1.63         3.09          430

               micro        46.81         2.47         4.70          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        15.00         0.95         1.79          316
                 NEU         0.00         0.00         0.00           25

               micro        15.00         0.62         1.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        15.00         0.95         1.79          316
                 NEU         0.00         0.00         0.00           25

               micro        15.00         0.62         1.19          483

Train epoch 17: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.23it/s]
Evaluate epoch 18: 100%|████████████████████████| 40/40 [00:04<00:00,  9.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        62.86         4.79         8.91          459
                   t        54.17         3.02         5.73          430

               micro        59.32         3.94         7.38          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        45.45         3.16         5.92          316
                 NEU         0.00         0.00         0.00           25

               micro        45.45         2.07         3.96          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        36.36         2.53         4.73          316
                 NEU         0.00         0.00         0.00           25

               micro        36.36         1.66         3.17          483

Train epoch 18: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.36it/s]
Evaluate epoch 19: 100%|████████████████████████| 40/40 [00:04<00:00,  8.80it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.00         8.93        15.16          459
                   t        58.33         1.63         3.17          430

               micro        51.06         5.40         9.77          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        33.33         1.90         3.59          316
                 NEU         0.00         0.00         0.00           25

               micro        33.33         1.24         2.40          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        27.78         1.58         2.99          316
                 NEU         0.00         0.00         0.00           25

               micro        27.78         1.04         2.00          483

Train epoch 19: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 20: 100%|████████████████████████| 40/40 [00:04<00:00,  8.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        59.38         4.14         7.74          459
                   t        42.86         2.09         3.99          430

               micro        52.83         3.15         5.94          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        16.67         1.27         2.35          316
                 NEU         0.00         0.00         0.00           25

               micro        16.67         0.83         1.58          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        16.67         1.27         2.35          316
                 NEU         0.00         0.00         0.00           25

               micro        16.67         0.83         1.58          483

Train epoch 20: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 21: 100%|████████████████████████| 40/40 [00:04<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.72         8.28        14.15          459
                   t        42.86         0.70         1.37          430

               micro        48.24         4.61         8.42          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        40.00         0.63         1.25          316
                 NEU         0.00         0.00         0.00           25

               micro        40.00         0.41         0.82          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        40.00         0.63         1.25          316
                 NEU         0.00         0.00         0.00           25

               micro        40.00         0.41         0.82          483

Train epoch 21: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 22: 100%|████████████████████████| 40/40 [00:04<00:00,  8.44it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.52         7.41        12.95          459
                   t        44.00         2.56         4.84          430

               micro        49.45         5.06         9.18          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        23.33         2.22         4.05          316
                 NEU         0.00         0.00         0.00           25

               micro        23.33         1.45         2.73          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        23.33         2.22         4.05          316
                 NEU         0.00         0.00         0.00           25

               micro        23.33         1.45         2.73          483

Train epoch 22: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.12it/s]
Evaluate epoch 23: 100%|████████████████████████| 40/40 [00:04<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        35.06        11.76        17.62          459
                   t        38.71         2.79         5.21          430

               micro        35.68         7.42        12.29          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        19.15         2.85         4.96          316
                 NEU         0.00         0.00         0.00           25

               micro        19.15         1.86         3.40          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        12.77         1.90         3.31          316
                 NEU         0.00         0.00         0.00           25

               micro        12.77         1.24         2.26          483

Train epoch 23: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.13it/s]
Evaluate epoch 24: 100%|████████████████████████| 40/40 [00:04<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.25         7.19        12.55          459
                   t        70.00         1.63         3.18          430

               micro        51.95         4.50         8.28          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        25.00         0.63         1.23          316
                 NEU         0.00         0.00         0.00           25

               micro        25.00         0.41         0.81          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        25.00         0.63         1.23          316
                 NEU         0.00         0.00         0.00           25

               micro        25.00         0.41         0.81          483

Train epoch 24: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 25: 100%|████████████████████████| 40/40 [00:04<00:00,  8.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.33         8.50        14.21          459
                   t        42.86         2.09         3.99          430

               micro        43.24         5.40         9.60          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        17.24         1.58         2.90          316
                 NEU         0.00         0.00         0.00           25

               micro        17.24         1.04         1.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        10.34         0.95         1.74          316
                 NEU         0.00         0.00         0.00           25

               micro        10.34         0.62         1.17          483

Train epoch 25: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 26: 100%|████████████████████████| 40/40 [00:04<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.36         4.36         7.78          459
                   t         0.00         0.00         0.00          430

               micro        33.90         2.25         4.22          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 26: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 27: 100%|████████████████████████| 40/40 [00:04<00:00,  8.29it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.42         3.05         5.69          459
                   t        50.00         1.40         2.71          430

               micro        44.44         2.25         4.28          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 27: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.21it/s]
Evaluate epoch 28: 100%|████████████████████████| 40/40 [00:04<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.54         4.14         7.44          459
                   t        33.33         1.40         2.68          430

               micro        35.71         2.81         5.21          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 28: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.21it/s]
Evaluate epoch 29: 100%|████████████████████████| 40/40 [00:04<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.20         4.14         7.34          459
                   t         0.00         0.00         0.00          430

               micro        30.16         2.14         3.99          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 29: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 30: 100%|████████████████████████| 40/40 [00:04<00:00,  8.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.92         5.23         9.16          459
                   t        50.00         1.16         2.27          430

               micro        38.67         3.26         6.02          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 30: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.23it/s]
Evaluate epoch 31: 100%|████████████████████████| 40/40 [00:04<00:00,  8.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.81         5.66         9.89          459
                   t        56.25         2.09         4.04          430

               micro        42.17         3.94         7.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        50.00         0.95         1.86          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         0.62         1.23          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 31: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.24it/s]
Evaluate epoch 32: 100%|████████████████████████| 40/40 [00:04<00:00,  9.12it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        37.97         6.54        11.15          459
                   t        29.41         1.16         2.24          430

               micro        36.46         3.94         7.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 32: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.26it/s]
Evaluate epoch 33: 100%|████████████████████████| 40/40 [00:04<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        39.74         6.75        11.55          459
                   t        71.43         1.16         2.29          430

               micro        42.35         4.05         7.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 33: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 34: 100%|████████████████████████| 40/40 [00:04<00:00,  8.47it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        35.85         4.14         7.42          459
                   t        23.68         6.28         9.93          430

               micro        27.54         5.17         8.71          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        10.31         3.16         4.84          316
                 NEU         0.00         0.00         0.00           25

               micro        10.31         2.07         3.45          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS         5.15         1.58         2.42          316
                 NEU         0.00         0.00         0.00           25

               micro         5.15         1.04         1.72          483

Train epoch 34: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 35: 100%|████████████████████████| 40/40 [00:04<00:00,  8.38it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        35.40         8.71        13.99          459
                   t        54.55         1.40         2.72          430

               micro        37.10         5.17         9.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS       100.00         0.32         0.63          316
                 NEU         0.00         0.00         0.00           25

               micro       100.00         0.21         0.41          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 35: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 36: 100%|████████████████████████| 40/40 [00:04<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        50.00         1.16         2.27          430

               micro        50.00         0.56         1.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 36: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.13it/s]
Evaluate epoch 37: 100%|████████████████████████| 40/40 [00:04<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.72         9.59        15.66          459
                   t        40.87        10.93        17.25          430

               micro        41.74        10.24        16.44          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        37.78         5.38         9.42          316
                 NEU         0.00         0.00         0.00           25

               micro        37.78         3.52         6.44          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        37.78         5.38         9.42          316
                 NEU         0.00         0.00         0.00           25

               micro        37.78         3.52         6.44          483

Train epoch 37: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 38: 100%|████████████████████████| 40/40 [00:04<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        50.00         1.16         2.27          430

               micro        50.00         0.56         1.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 38: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 39: 100%|████████████████████████| 40/40 [00:04<00:00,  8.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         1.53         3.00          459
                   t        50.00         1.16         2.27          430

               micro        70.59         1.35         2.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 39: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 40: 100%|████████████████████████| 40/40 [00:04<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.62         4.36         7.98          459
                   t        50.00         1.16         2.27          430

               micro        48.08         2.81         5.31          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 40: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.21it/s]
Evaluate epoch 41: 100%|████████████████████████| 40/40 [00:04<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.88         6.54        11.47          459
                   t        83.33         1.16         2.29          430

               micro        50.00         3.94         7.30          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 41: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 42: 100%|████████████████████████| 40/40 [00:04<00:00,  8.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.33         5.66        10.02          459
                   t        50.00         1.16         2.27          430

               micro        44.29         3.49         6.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 42: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 43: 100%|████████████████████████| 40/40 [00:04<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.16         3.05         5.71          459
                   t        50.00         1.16         2.27          430

               micro        46.34         2.14         4.09          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 43: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 44: 100%|████████████████████████| 40/40 [00:04<00:00,  8.24it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.31         2.40         4.54          459
                   t        50.00         1.16         2.27          430

               micro        44.44         1.80         3.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 44: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 45: 100%|████████████████████████| 40/40 [00:04<00:00,  8.44it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.88         6.54        11.47          459
                   t        83.33         1.16         2.29          430

               micro        50.00         3.94         7.30          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 45: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 46: 100%|████████████████████████| 40/40 [00:04<00:00,  8.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        57.69         6.98        12.45          430

               micro        53.49         7.76        13.56          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        34.55         6.01        10.24          316
                 NEU         0.00         0.00         0.00           25

               micro        34.55         3.93         7.06          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        34.55         6.01        10.24          316
                 NEU         0.00         0.00         0.00           25

               micro        34.55         3.93         7.06          483

Train epoch 46: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 47: 100%|████████████████████████| 40/40 [00:04<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.43         3.92         7.29          459
                   t        83.33         1.16         2.29          430

               micro        56.10         2.59         4.95          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 47: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.30it/s]
Evaluate epoch 48: 100%|████████████████████████| 40/40 [00:04<00:00,  9.02it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        33.65        15.25        20.99          459
                   t        31.48        11.86        17.23          430

               micro        32.70        13.61        19.22          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        20.22         5.70         8.89          316
                 NEU         0.00         0.00         0.00           25

               micro        20.22         3.73         6.29          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        20.22         5.70         8.89          316
                 NEU         0.00         0.00         0.00           25

               micro        20.22         3.73         6.29          483

Train epoch 48: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.35it/s]
Evaluate epoch 49: 100%|████████████████████████| 40/40 [00:04<00:00,  9.02it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.62        10.89        17.73          459
                   t        30.67        16.98        21.86          430

               micro        35.86        13.84        19.97          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        21.95         5.70         9.05          316
                 NEU         0.00         0.00         0.00           25

               micro        16.36         3.73         6.07          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        19.51         5.06         8.04          316
                 NEU         0.00         0.00         0.00           25

               micro        14.55         3.31         5.40          483

Train epoch 49: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.28it/s]
Evaluate epoch 50: 100%|████████████████████████| 40/40 [00:04<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.16         3.05         5.71          459
                   t        50.00         1.16         2.27          430

               micro        46.34         2.14         4.09          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 50: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 51: 100%|████████████████████████| 40/40 [00:04<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.16         3.05         5.71          459
                   t        45.45         1.16         2.27          430

               micro        45.24         2.14         4.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 51: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Train epoch 52: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 53: 100%|████████████████████████| 40/40 [00:04<00:00,  8.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.77         6.54        11.95          459
                   t        70.00         3.26         6.22          430

               micro        69.84         4.95         9.24          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        57.89         3.48         6.57          316
                 NEU         0.00         0.00         0.00           25

               micro        57.89         2.28         4.38          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        57.89         3.48         6.57          316
                 NEU         0.00         0.00         0.00           25

               micro        57.89         2.28         4.38          483

Train epoch 53: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.13it/s]
Evaluate epoch 54: 100%|████████████████████████| 40/40 [00:04<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.35         7.84        13.24          459
                   t         0.00         0.00         0.00          430

               micro        42.35         4.05         7.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        83.33         1.58         3.11          316
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 54: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 55: 100%|████████████████████████| 40/40 [00:04<00:00,  8.79it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.81         8.93        15.10          459
                   t        54.43        10.00        16.90          430

               micro        51.53         9.45        15.97          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        29.82         5.38         9.12          316
                 NEU         0.00         0.00         0.00           25

               micro        29.82         3.52         6.30          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        28.07         5.06         8.58          316
                 NEU         0.00         0.00         0.00           25

               micro        28.07         3.31         5.93          483

Train epoch 55: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.24it/s]
Evaluate epoch 56: 100%|████████████████████████| 40/40 [00:04<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.64         5.23         9.34          459
                   t        62.39        15.81        25.23          430

               micro        56.10        10.35        17.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        55.17         5.06         9.28          316
                 NEU         0.00         0.00         0.00           25

               micro        55.17         3.31         6.25          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        55.17         5.06         9.28          316
                 NEU         0.00         0.00         0.00           25

               micro        55.17         3.31         6.25          483

Train epoch 56: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.12it/s]
Evaluate epoch 57: 100%|████████████████████████| 40/40 [00:04<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.85         5.88        10.80          459
                   t        40.38        20.00        26.75          430

               micro        44.49        12.71        19.77          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        34.69         5.38         9.32          316
                 NEU         0.00         0.00         0.00           25

               micro        34.69         3.52         6.39          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        32.65         5.06         8.77          316
                 NEU         0.00         0.00         0.00           25

               micro        32.65         3.31         6.02          483

Train epoch 57: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 58: 100%|████████████████████████| 40/40 [00:04<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.78         6.54        11.41          459
                   t        59.46        10.23        17.46          430

               micro        52.48         8.32        14.37          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        50.00         5.06         9.20          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         3.31         6.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        50.00         5.06         9.20          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         3.31         6.21          483

Train epoch 58: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 59: 100%|████████████████████████| 40/40 [00:04<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         2.61         5.10          459
                   t        62.96         7.91        14.05          430

               micro        69.70         5.17         9.63          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        85.71         1.90         3.72          316
                 NEU         0.00         0.00         0.00           25

               micro        75.00         1.24         2.44          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        85.71         1.90         3.72          316
                 NEU         0.00         0.00         0.00           25

               micro        75.00         1.24         2.44          483

Train epoch 59: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 60: 100%|████████████████████████| 40/40 [00:04<00:00,  8.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        58.62         3.70         6.97          459
                   t        57.55        14.19        22.76          430

               micro        57.78         8.77        15.23          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        90.00         2.85         5.52          316
                 NEU         0.00         0.00         0.00           25

               micro        81.82         1.86         3.64          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        90.00         2.85         5.52          316
                 NEU         0.00         0.00         0.00           25

               micro        81.82         1.86         3.64          483

Train epoch 60: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 61: 100%|████████████████████████| 40/40 [00:04<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.85         6.32        10.94          459
                   t        70.83         3.95         7.49          430

               micro        48.42         5.17         9.35          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        57.89         3.48         6.57          316
                 NEU         0.00         0.00         0.00           25

               micro        57.89         2.28         4.38          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        47.37         2.85         5.37          316
                 NEU         0.00         0.00         0.00           25

               micro        47.37         1.86         3.59          483

Train epoch 61: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 62: 100%|████████████████████████| 40/40 [00:04<00:00,  8.15it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.30         7.63        13.01          459
                   t        59.62         7.21        12.86          430

               micro        50.38         7.42        12.94          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        26.67         2.53         4.62          316
                 NEU         0.00         0.00         0.00           25

               micro        26.67         1.66         3.12          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        26.67         2.53         4.62          316
                 NEU         0.00         0.00         0.00           25

               micro        26.67         1.66         3.12          483

Train epoch 62: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 63: 100%|████████████████████████| 40/40 [00:04<00:00,  8.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.76         6.97        12.17          459
                   t        55.08        15.12        23.72          430

               micro        52.43        10.91        18.06          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.03         5.06         9.01          316
                 NEU         0.00         0.00         0.00           25

               micro        41.03         3.31         6.13          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.03         5.06         9.01          316
                 NEU         0.00         0.00         0.00           25

               micro        41.03         3.31         6.13          483

Train epoch 63: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.13it/s]
Evaluate epoch 64: 100%|████████████████████████| 40/40 [00:04<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        68.97         4.36         8.20          459
                   t        53.33        14.88        23.27          430

               micro        56.38         9.45        16.18          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        81.82         2.85         5.50          316
                 NEU         0.00         0.00         0.00           25

               micro        60.00         1.86         3.61          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        81.82         2.85         5.50          316
                 NEU         0.00         0.00         0.00           25

               micro        60.00         1.86         3.61          483

Train epoch 64: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 65: 100%|████████████████████████| 40/40 [00:04<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.92         6.54        12.05          459
                   t        62.28        16.51        26.10          430

               micro        66.01        11.36        19.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        56.52         4.11         7.67          316
                 NEU         0.00         0.00         0.00           25

               micro        56.52         2.69         5.14          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        56.52         4.11         7.67          316
                 NEU         0.00         0.00         0.00           25

               micro        56.52         2.69         5.14          483

Train epoch 65: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.12it/s]
Evaluate epoch 66: 100%|████████████████████████| 40/40 [00:04<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        95.24         4.36         8.33          459
                   t        69.07        15.58        25.43          430

               micro        73.73         9.79        17.28          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        81.25         4.11         7.83          316
                 NEU         0.00         0.00         0.00           25

               micro        81.25         2.69         5.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        81.25         4.11         7.83          316
                 NEU         0.00         0.00         0.00           25

               micro        81.25         2.69         5.21          483

Train epoch 66: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.09it/s]
Evaluate epoch 67: 100%|████████████████████████| 40/40 [00:05<00:00,  7.83it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        53.73         7.84        13.69          459
                   t        69.57        14.88        24.52          430

               micro        62.89        11.25        19.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        48.39         4.75         8.65          316
                 NEU         0.00         0.00         0.00           25

               micro        48.39         3.11         5.84          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        48.39         4.75         8.65          316
                 NEU         0.00         0.00         0.00           25

               micro        48.39         3.11         5.84          483

Train epoch 67: 100%|███████████████████████████| 74/74 [00:24<00:00,  3.04it/s]
Evaluate epoch 68: 100%|████████████████████████| 40/40 [00:04<00:00,  8.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        54.39         6.75        12.02          459
                   t        59.46        15.35        24.40          430

               micro        57.74        10.91        18.35          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        60.00         4.75         8.80          316
                 NEU         0.00         0.00         0.00           25

               micro        60.00         3.11         5.91          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        60.00         4.75         8.80          316
                 NEU         0.00         0.00         0.00           25

               micro        60.00         3.11         5.91          483

Train epoch 68: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 69: 100%|████████████████████████| 40/40 [00:04<00:00,  8.39it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        92.59         5.45        10.29          459
                   t        61.39        14.42        23.35          430

               micro        67.97         9.79        17.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        75.00         4.75         8.93          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         3.11         5.85          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        75.00         4.75         8.93          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         3.11         5.85          483

Train epoch 69: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 70: 100%|████████████████████████| 40/40 [00:04<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        87.50         3.05         5.89          459
                   t        56.86        13.49        21.80          430

               micro        61.02         8.10        14.30          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        68.75         3.48         6.63          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         2.28         4.36          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        68.75         3.48         6.63          316
                 NEU         0.00         0.00         0.00           25

               micro        50.00         2.28         4.36          483

Train epoch 70: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 71: 100%|████████████████████████| 40/40 [00:04<00:00,  8.38it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.94         5.88        10.59          459
                   t        66.67        14.42        23.71          430

               micro        61.81        10.01        17.23          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        45.83         3.48         6.47          316
                 NEU         0.00         0.00         0.00           25

               micro        42.31         2.28         4.32          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.67         3.16         5.88          316
                 NEU         0.00         0.00         0.00           25

               micro        38.46         2.07         3.93          483

Train epoch 71: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 72: 100%|████████████████████████| 40/40 [00:04<00:00,  8.13it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.55         8.71        14.47          459
                   t        63.89        16.05        25.65          430

               micro        53.96        12.26        19.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        33.33         6.65        11.08          316
                 NEU         0.00         0.00         0.00           25

               micro        25.00         4.35         7.41          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        33.33         6.65        11.08          316
                 NEU         0.00         0.00         0.00           25

               micro        25.00         4.35         7.41          483

Train epoch 72: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 73: 100%|████████████████████████| 40/40 [00:04<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.48        10.46        17.20          459
                   t        68.82        14.88        24.47          430

               micro        58.33        12.60        20.72          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        44.90         6.96        12.05          316
                 NEU         0.00         0.00         0.00           25

               micro        40.74         4.55         8.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        42.86         6.65        11.51          316
                 NEU         0.00         0.00         0.00           25

               micro        38.89         4.35         7.82          483

Train epoch 73: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 74: 100%|████████████████████████| 40/40 [00:04<00:00,  8.20it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.44        13.07        20.20          459
                   t        70.59        16.74        27.07          430

               micro        55.70        14.85        23.45          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        39.13         8.54        14.03          316
                 NEU         0.00         0.00         0.00           25

               micro        39.13         5.59         9.78          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        39.13         8.54        14.03          316
                 NEU         0.00         0.00         0.00           25

               micro        39.13         5.59         9.78          483

Train epoch 74: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 75: 100%|████████████████████████| 40/40 [00:04<00:00,  8.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.08         8.28        14.87          459
                   t        68.87        16.98        27.24          430

               micro        70.25        12.49        21.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        58.82         6.33        11.43          316
                 NEU         0.00         0.00         0.00           25

               micro        40.82         4.14         7.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        58.82         6.33        11.43          316
                 NEU         0.00         0.00         0.00           25

               micro        40.82         4.14         7.52          483

Train epoch 75: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 76: 100%|████████████████████████| 40/40 [00:04<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.44         3.49         6.46          459
                   t        49.32         8.37        14.31          430

               micro        47.71         5.85        10.42          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00        142.0
                 POS         0.00         0.00         0.00        316.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 76: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 77: 100%|████████████████████████| 40/40 [00:04<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        59.52         5.45         9.98          459
                   t        68.54        14.19        23.51          430

               micro        65.65         9.67        16.86          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        44.00         3.48         6.45          316
                 NEU         0.00         0.00         0.00           25

               micro        28.95         2.28         4.22          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        44.00         3.48         6.45          316
                 NEU         0.00         0.00         0.00           25

               micro        28.95         2.28         4.22          483

Train epoch 77: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.23it/s]
Evaluate epoch 78: 100%|████████████████████████| 40/40 [00:04<00:00,  8.91it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        57.35         8.50        14.80          459
                   t        52.67        16.05        24.60          430

               micro        54.27        12.15        19.85          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        55.56         4.75         8.75          316
                 NEU         0.00         0.00         0.00           25

               micro        36.59         3.11         5.73          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        55.56         4.75         8.75          316
                 NEU         0.00         0.00         0.00           25

               micro        36.59         3.11         5.73          483

Train epoch 78: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.29it/s]
Evaluate epoch 79: 100%|████████████████████████| 40/40 [00:04<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        55.67        11.76        19.42          459
                   t        67.05        13.72        22.78          430

               micro        61.08        12.71        21.04          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.43         9.18        15.03          316
                 NEU         0.00         0.00         0.00           25

               micro        41.43         6.00        10.49          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        40.00         8.86        14.51          316
                 NEU         0.00         0.00         0.00           25

               micro        40.00         5.80        10.13          483

Train epoch 79: 100%|███████████████████████████| 74/74 [00:22<00:00,  3.24it/s]
Evaluate epoch 80: 100%|████████████████████████| 40/40 [00:04<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.78        14.16        22.15          459
                   t        66.96        17.91        28.26          430

               micro        58.44        15.97        25.09          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        35.71        11.08        16.91          316
                 NEU         0.00         0.00         0.00           25

               micro        33.98         7.25        11.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        35.71        11.08        16.91          316
                 NEU         0.00         0.00         0.00           25

               micro        33.98         7.25        11.95          483

Train epoch 80: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 81: 100%|████████████████████████| 40/40 [00:04<00:00,  8.14it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.72        16.56        24.72          459
                   t        55.48        18.84        28.12          430

               micro        51.99        17.66        26.36          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         8.00         2.82         4.17          142
                 POS        33.72         9.18        14.43          316
                 NEU         0.00         0.00         0.00           25

               micro        24.26         6.83        10.66          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         8.00         2.82         4.17          142
                 POS        33.72         9.18        14.43          316
                 NEU         0.00         0.00         0.00           25

               micro        24.26         6.83        10.66          483

Train epoch 81: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 82: 100%|████████████████████████| 40/40 [00:05<00:00,  7.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.67        14.81        23.49          459
                   t        64.86        16.74        26.62          430

               micro        60.61        15.75        25.00          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG       100.00         0.70         1.40          142
                 POS        38.37        10.44        16.42          316
                 NEU         0.00         0.00         0.00           25

               micro        39.08         7.04        11.93          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG       100.00         0.70         1.40          142
                 POS        38.37        10.44        16.42          316
                 NEU         0.00         0.00         0.00           25

               micro        39.08         7.04        11.93          483

Train epoch 82: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 83: 100%|████████████████████████| 40/40 [00:04<00:00,  8.10it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        80.77         4.58         8.66          459
                   t        62.69         9.77        16.90          430

               micro        67.74         7.09        12.83          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        62.50         1.58         3.09          316
                 NEU         0.00         0.00         0.00           25

               micro        62.50         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        62.50         1.58         3.09          316
                 NEU         0.00         0.00         0.00           25

               micro        62.50         1.04         2.04          483

Train epoch 83: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 84: 100%|████████████████████████| 40/40 [00:04<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        55.00        11.98        19.68          459
                   t        65.38        15.81        25.47          430

               micro        60.29        13.84        22.51          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        47.76        10.13        16.71          316
                 NEU         0.00         0.00         0.00           25

               micro        45.07         6.63        11.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        47.76        10.13        16.71          316
                 NEU         0.00         0.00         0.00           25

               micro        45.07         6.63        11.55          483

Train epoch 84: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 85: 100%|████████████████████████| 40/40 [00:04<00:00,  8.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.57        10.68        17.44          459
                   t        54.31        14.65        23.08          430

               micro        51.14        12.60        20.22          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00           25
                 POS        35.21         7.91        12.92          316
                 NEG         0.00         0.00         0.00          142

               micro        28.74         5.18         8.77          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00           25
                 POS        35.21         7.91        12.92          316
                 NEG         0.00         0.00         0.00          142

               micro        28.74         5.18         8.77          483

Train epoch 85: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 86: 100%|████████████████████████| 40/40 [00:04<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.00        11.33        18.60          459
                   t        62.07        16.74        26.37          430

               micro        57.41        13.95        22.44          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        45.95        10.76        17.44          316
                 NEU         0.00         0.00         0.00           25

               micro        40.96         7.04        12.01          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        45.95        10.76        17.44          316
                 NEU         0.00         0.00         0.00           25

               micro        40.96         7.04        12.01          483

Train epoch 86: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 87: 100%|████████████████████████| 40/40 [00:04<00:00,  8.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        60.44        11.98        20.00          459
                   t        65.14        16.51        26.35          430

               micro        63.00        14.17        23.14          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        52.54         9.81        16.53          316
                 NEU         0.00         0.00         0.00           25

               micro        51.67         6.42        11.42          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        52.54         9.81        16.53          316
                 NEU         0.00         0.00         0.00           25

               micro        51.67         6.42        11.42          483

Train epoch 87: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 88: 100%|████████████████████████| 40/40 [00:04<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        55.26        13.73        21.99          459
                   t        61.60        17.91        27.75          430

               micro        58.58        15.75        24.82          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        44.09        12.97        20.05          316
                 NEU         0.00         0.00         0.00           25

               micro        44.09         8.49        14.24          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        43.01        12.66        19.56          316
                 NEU         0.00         0.00         0.00           25

               micro        43.01         8.28        13.89          483

Train epoch 88: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.18it/s]
Evaluate epoch 89: 100%|████████████████████████| 40/40 [00:04<00:00,  8.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        54.33        15.03        23.55          459
                   t        59.85        18.37        28.11          430

               micro        57.14        16.65        25.78          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        17.39         2.82         4.85          142
                 POS        39.00        12.34        18.75          316
                 NEU         0.00         0.00         0.00           25

               micro        34.96         8.90        14.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        17.39         2.82         4.85          142
                 POS        39.00        12.34        18.75          316
                 NEU         0.00         0.00         0.00           25

               micro        34.96         8.90        14.19          483

Train epoch 89: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 90: 100%|████████████████████████| 40/40 [00:04<00:00,  8.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.86        16.12        24.71          459
                   t        68.87        16.98        27.24          430

               micro        59.76        16.54        25.90          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         8.70         1.41         2.42          142
                 POS        40.96        10.76        17.04          316
                 NEU         0.00         0.00         0.00           25

               micro        33.96         7.45        12.22          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         8.70         1.41         2.42          142
                 POS        40.96        10.76        17.04          316
                 NEU         0.00         0.00         0.00           25

               micro        33.96         7.45        12.22          483

Train epoch 90: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 91: 100%|████████████████████████| 40/40 [00:04<00:00,  8.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        60.24        10.89        18.45          459
                   t        64.89        14.19        23.28          430

               micro        62.71        12.49        20.83          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.38         7.59        12.83          316
                 NEU         0.00         0.00         0.00           25

               micro        35.82         4.97         8.73          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        41.38         7.59        12.83          316
                 NEU         0.00         0.00         0.00           25

               micro        35.82         4.97         8.73          483

Train epoch 91: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.14it/s]
Evaluate epoch 92: 100%|████████████████████████| 40/40 [00:04<00:00,  8.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        64.20        11.33        19.26          459
                   t        73.12        15.81        26.00          430

               micro        68.97        13.50        22.58          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         9.38         2.11         3.45          142
                 POS        55.56         9.49        16.22          316
                 NEU         0.00         0.00         0.00           25

               micro        38.37         6.83        11.60          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         9.38         2.11         3.45          142
                 POS        55.56         9.49        16.22          316
                 NEU         0.00         0.00         0.00           25

               micro        38.37         6.83        11.60          483

Train epoch 92: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.20it/s]
Evaluate epoch 93: 100%|████████████████████████| 40/40 [00:04<00:00,  8.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.98        16.99        25.49          459
                   t        52.03        17.91        26.64          430

               micro        51.50        17.44        26.05          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00           25
                 POS        32.41        11.08        16.51          316
                 NEG        13.04         2.11         3.64          142

               micro        26.57         7.87        12.14          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEU         0.00         0.00         0.00           25
                 POS        31.48        10.76        16.04          316
                 NEG        13.04         2.11         3.64          142

               micro        25.87         7.66        11.82          483

Train epoch 93: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 94: 100%|████████████████████████| 40/40 [00:04<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.23        18.08        26.73          459
                   t        56.77        20.47        30.09          430

               micro        53.94        19.24        28.36          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         4.55         0.70         1.22          142
                 POS        35.04        12.97        18.94          316
                 NEU         0.00         0.00         0.00           25

               micro        30.00         8.70        13.48          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         4.55         0.70         1.22          142
                 POS        35.04        12.97        18.94          316
                 NEU         0.00         0.00         0.00           25

               micro        30.00         8.70        13.48          483

Train epoch 94: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 95: 100%|████████████████████████| 40/40 [00:04<00:00,  8.04it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        54.17        16.99        25.87          459
                   t        60.77        18.37        28.21          430

               micro        57.30        17.66        27.00          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        11.11         0.70         1.32          142
                 POS        42.27        12.97        19.85          316
                 NEU         0.00         0.00         0.00           25

               micro        39.62         8.70        14.26          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        11.11         0.70         1.32          142
                 POS        41.24        12.66        19.37          316
                 NEU         0.00         0.00         0.00           25

               micro        38.68         8.49        13.92          483

Train epoch 95: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 96: 100%|████████████████████████| 40/40 [00:04<00:00,  8.29it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        58.77        14.60        23.39          459
                   t        65.35        19.30        29.80          430

               micro        62.24        16.87        26.55          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        11.11         1.41         2.50          142
                 POS        47.62        12.66        20.00          316
                 NEU         0.00         0.00         0.00           25

               micro        41.18         8.70        14.36          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        11.11         1.41         2.50          142
                 POS        47.62        12.66        20.00          316
                 NEU         0.00         0.00         0.00           25

               micro        41.18         8.70        14.36          483

Train epoch 96: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.16it/s]
Evaluate epoch 97: 100%|████████████████████████| 40/40 [00:04<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.71        13.94        21.37          459
                   t        69.30        18.37        29.04          430

               micro        56.30        16.09        25.02          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        35.87        10.44        16.18          316
                 NEU         0.00         0.00         0.00           25

               micro        30.84         6.83        11.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        35.87        10.44        16.18          316
                 NEU         0.00         0.00         0.00           25

               micro        30.84         6.83        11.19          483

Train epoch 97: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.15it/s]
Evaluate epoch 98: 100%|████████████████████████| 40/40 [00:04<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.14        13.94        22.34          459
                   t        55.10        18.84        28.08          430

               micro        55.56        16.31        25.22          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        37.50         8.54        13.92          316
                 NEU         0.00         0.00         0.00           25

               micro        32.53         5.59         9.54          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG         0.00         0.00         0.00          142
                 POS        37.50         8.54        13.92          316
                 NEU         0.00         0.00         0.00           25

               micro        32.53         5.59         9.54          483

Train epoch 98: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.19it/s]
Evaluate epoch 99: 100%|████████████████████████| 40/40 [00:04<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        55.56        17.43        26.53          459
                   t        67.80        18.60        29.20          430

               micro        61.07        18.00        27.80          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        15.62         3.52         5.75          142
                 POS        44.21        13.29        20.44          316
                 NEU         0.00         0.00         0.00           25

               micro        37.01         9.73        15.41          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        15.62         3.52         5.75          142
                 POS        44.21        13.29        20.44          316
                 NEU         0.00         0.00         0.00           25

               micro        37.01         9.73        15.41          483

Train epoch 99: 100%|███████████████████████████| 74/74 [00:23<00:00,  3.17it/s]
Evaluate epoch 100: 100%|███████████████████████| 40/40 [00:04<00:00,  8.22it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.50        17.65        25.88          459
                   t        69.23        16.74        26.97          430

               micro        56.46        17.21        26.38          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 NEG        15.62         3.52         5.75          142
                 POS        46.27         9.81        16.19          316
                 NEU         0.00         0.00         0.00           25

               micro        36.36         7.45        12.37          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 NEG        15.62         3.52         5.75          142
                 POS        46.27         9.81        16.19          316
                 NEU         0.00         0.00         0.00           25

               micro        36.36         7.45        12.37          483


Best F1 score: 40 at epoch -1