🚀 Starting training with parameters:
  - Model: microsoft/deberta-v3-base
  - emb_dim: 768
  - hidden_dim: 384 (bidirectional → 768)
  - deberta_feature_dim: 768
  - gcn_dim: 768
  - mem_dim: 768
============================================================
2025-12-29 07:06:45.132932: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1766992005.153426     380 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1766992005.159352     380 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
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
Parse dataset 'train': 100%|████████████████| 592/592 [00:00<00:00, 1896.54it/s]
Parse dataset 'test': 100%|█████████████████| 320/320 [00:00<00:00, 1876.49it/s]
    15res    8
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing D2E2SModel: ['mask_predictions.classifier.weight', 'lm_predictions.lm_head.bias', 'lm_predictions.lm_head.dense.weight', 'lm_predictions.lm_head.LayerNorm.bias', 'mask_predictions.LayerNorm.weight', 'lm_predictions.lm_head.LayerNorm.weight', 'lm_predictions.lm_head.dense.bias', 'mask_predictions.LayerNorm.bias', 'mask_predictions.dense.bias', 'mask_predictions.dense.weight', 'deberta.embeddings.position_embeddings.weight', 'mask_predictions.classifier.bias']
- This IS expected if you are initializing D2E2SModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing D2E2SModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['lstm.weight_hh_l1_reverse', 'TIN.residual_layer1.3.weight', 'TIN.residual_layer2.2.bias', 'lstm.weight_hh_l0_reverse', 'fc.bias', 'TIN.feature_fusion.2.bias', 'TIN.GatedGCN.conv3.bias', 'TIN.residual_layer3.0.weight', 'TIN.lstm.bias_hh_l1_reverse', 'TIN.lstm.bias_hh_l1', 'TIN.lstm.bias_ih_l1_reverse', 'TIN.residual_layer2.3.weight', 'attention_layer.w_value.bias', 'Syn_gcn.W.0.bias', 'TIN.residual_layer4.3.bias', 'senti_classifier.weight', 'lstm.bias_ih_l0_reverse', 'TIN.lstm.weight_hh_l0', 'lstm.bias_hh_l1', 'TIN.GatedGCN.conv2.lin_l.bias', 'Syn_gcn.W.0.weight', 'TIN.residual_layer2.2.weight', 'TIN.GatedGCN.conv3.lin_l.weight', 'TIN.residual_layer3.2.weight', 'lstm.weight_ih_l0_reverse', 'TIN.GatedGCN.conv1.lin_l.weight', 'Sem_gcn.attn.linears.0.bias', 'Sem_gcn.W.0.bias', 'attention_layer.v.weight', 'TIN.residual_layer1.2.weight', 'entity_classifier.bias', 'lstm.bias_hh_l1_reverse', 'TIN.GatedGCN.conv2.att', 'TIN.lstm.bias_ih_l1', 'TIN.lstm.weight_hh_l1', 'TIN.GatedGCN.conv1.lin_r.bias', 'TIN.GatedGCN.conv2.lin_r.bias', 'TIN.residual_layer3.2.bias', 'TIN.lstm.weight_ih_l0', 'TIN.residual_layer4.2.bias', 'TIN.feature_fusion.0.weight', 'Sem_gcn.attn.linears.1.bias', 'TIN.GatedGCN.conv3.lin_r.bias', 'lstm.bias_ih_l1', 'Syn_gcn.W.1.weight', 'lstm.bias_ih_l1_reverse', 'TIN.feature_fusion.3.bias', 'TIN.GatedGCN.conv2.bias', 'TIN.GatedGCN.conv3.lin_r.weight', 'TIN.GatedGCN.conv1.bias', 'senti_classifier.bias', 'lstm.bias_hh_l0', 'lstm.weight_ih_l0', 'TIN.GatedGCN.conv1.lin_r.weight', 'fc.weight', 'Sem_gcn.W.0.weight', 'TIN.GatedGCN.conv3.att', 'TIN.residual_layer2.0.weight', 'lstm.bias_hh_l0_reverse', 'attention_layer.w_query.bias', 'lstm.weight_hh_l0', 'TIN.residual_layer2.3.bias', 'TIN.residual_layer4.2.weight', 'TIN.residual_layer4.3.weight', 'TIN.feature_fusion.0.bias', 'Syn_gcn.W.1.bias', 'TIN.feature_fusion.2.weight', 'TIN.GatedGCN.conv2.lin_l.weight', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.feature_fusion.3.weight', 'TIN.residual_layer1.0.weight', 'TIN.lstm.bias_hh_l0_reverse', 'attention_layer.w_value.weight', 'lstm.weight_ih_l1_reverse', 'TIN.residual_layer2.0.bias', 'TIN.GatedGCN.conv1.att', 'TIN.residual_layer3.0.bias', 'entity_classifier.weight', 'attention_layer.linear_q.bias', 'TIN.residual_layer1.0.bias', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer4.0.weight', 'deberta.embeddings.position_ids', 'TIN.GatedGCN.conv1.lin_l.bias', 'TIN.GatedGCN.conv2.lin_r.weight', 'lstm.bias_ih_l0', 'lstm.weight_ih_l1', 'TIN.residual_layer3.3.bias', 'TIN.GatedGCN.conv3.lin_l.bias', 'TIN.lstm.weight_ih_l1', 'attention_layer.linear_q.weight', 'TIN.lstm.weight_hh_l1_reverse', 'Sem_gcn.attn.linears.0.weight', 'TIN.residual_layer3.3.weight', 'TIN.lstm.bias_ih_l0_reverse', 'TIN.residual_layer4.0.bias', 'TIN.residual_layer1.3.bias', 'TIN.lstm.weight_ih_l0_reverse', 'attention_layer.w_query.weight', 'size_embeddings.weight', 'TIN.lstm.bias_ih_l0', 'TIN.residual_layer1.2.bias', 'Sem_gcn.W.1.weight', 'TIN.lstm.bias_hh_l0', 'lstm.weight_hh_l1', 'TIN.lstm.weight_hh_l0_reverse', 'Sem_gcn.W.1.bias']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Train epoch 0: 100%|████████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 1: 100%|█████████████████████████| 27/27 [00:03<00:00,  8.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 1: 100%|████████████████████████████| 49/49 [00:16<00:00,  2.99it/s]
Evaluate epoch 2: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 2: 100%|████████████████████████████| 49/49 [00:16<00:00,  3.00it/s]
Evaluate epoch 3: 100%|█████████████████████████| 27/27 [00:05<00:00,  4.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 3: 100%|████████████████████████████| 49/49 [00:16<00:00,  2.88it/s]
Evaluate epoch 4: 100%|█████████████████████████| 27/27 [00:03<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.74         3.70         6.84          459
                   t         0.00         0.00         0.00          430

               micro        44.74         1.91         3.67          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.35         3.48         6.29          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.35         2.28         4.26          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 4: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 5: 100%|█████████████████████████| 27/27 [00:03<00:00,  8.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        36.73         4.19         7.52          430

               micro        36.73         2.02         3.84          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         1.58         2.98          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         1.04         1.99          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 5: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 6: 100%|█████████████████████████| 27/27 [00:03<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        66.67         4.19         7.88          430

               micro        66.67         2.02         3.93          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 6: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 7: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.75it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.79        17.65        22.95          459
                   t        30.47        52.79        38.64          430

               micro        31.05        34.65        32.75          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        17.20        17.09        17.14          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        17.20        11.18        13.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        11.15        11.08        11.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.15         7.25         8.78          483

Train epoch 7: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.82it/s]
Evaluate epoch 8: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.92it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        91.67         2.40         4.67          459
                   t        55.31        47.21        50.94          430

               micro        56.46        24.07        33.75          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        50.00         2.22         4.24          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        50.00         1.45         2.82          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        50.00         2.22         4.24          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        50.00         1.45         2.82          483

Train epoch 8: 100%|████████████████████████████| 49/49 [00:16<00:00,  2.88it/s]
Evaluate epoch 9: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.09        35.08        40.91          459
                   t        43.30        57.91        49.55          430

               micro        45.40        46.12        45.76          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.36        32.28        29.02          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.29        21.12        23.42          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        26.10        31.96        28.73          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.03        20.91        23.19          483

Train epoch 9: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 10: 100%|████████████████████████| 27/27 [00:03<00:00,  7.82it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        91.43         6.97        12.96          459
                   t        56.47        47.67        51.70          430

               micro        59.55        26.66        36.83          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        58.14         7.91        13.93          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        58.14         5.18         9.51          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        58.14         7.91        13.93          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        58.14         5.18         9.51          483

Train epoch 10: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 11: 100%|████████████████████████| 27/27 [00:03<00:00,  7.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        59.59        44.01        50.63          459
                   t        52.09        52.09        52.09          430

               micro        55.40        47.92        51.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.29        32.59        32.44          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.29        21.33        25.69          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.29        32.59        32.44          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.29        21.33        25.69          483

Train epoch 11: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.80it/s]
Evaluate epoch 12: 100%|████████████████████████| 27/27 [00:03<00:00,  7.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.90        59.26        48.40          459
                   t        58.64        52.09        55.17          430

               micro        47.37        55.79        51.24          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        21.06        40.19        27.64          316
                 NEG        42.86         2.11         4.03          142
                 NEU         0.00         0.00         0.00           25

               micro        21.31        26.92        23.79          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        20.90        39.87        27.42          316
                 NEG        42.86         2.11         4.03          142
                 NEU         0.00         0.00         0.00           25

               micro        21.15        26.71        23.60          483

Train epoch 12: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 13: 100%|████████████████████████| 27/27 [00:03<00:00,  7.77it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        57.21        54.47        55.80          459
                   t        71.11        44.65        54.86          430

               micro        62.52        49.72        55.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        24.73        36.39        29.45          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.68        23.81        24.24          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        24.30        35.76        28.94          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.25        23.40        23.81          483

Train epoch 13: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 14: 100%|████████████████████████| 27/27 [00:03<00:00,  8.00it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.97        46.19        55.64          459
                   t        61.27        53.72        57.25          430

               micro        65.15        49.83        56.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        42.55        37.03        39.59          316
                 NEG        31.25         3.52         6.33          142
                 NEU         0.00         0.00         0.00           25

               micro        41.92        25.26        31.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        42.55        37.03        39.59          316
                 NEG        31.25         3.52         6.33          142
                 NEU         0.00         0.00         0.00           25

               micro        41.92        25.26        31.52          483

Train epoch 14: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 15: 100%|████████████████████████| 27/27 [00:03<00:00,  7.87it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.08        49.67        59.14          459
                   t        55.67        60.47        57.97          430

               micro        62.64        54.89        58.51          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        41.64        41.77        41.71          316
                 NEG        40.00         5.63         9.88          142
                 NEU         0.00         0.00         0.00           25

               micro        41.54        28.99        34.15          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        41.64        41.77        41.71          316
                 NEG        40.00         5.63         9.88          142
                 NEU         0.00         0.00         0.00           25

               micro        41.54        28.99        34.15          483

Train epoch 15: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 16: 100%|████████████████████████| 27/27 [00:03<00:00,  7.94it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        64.74        44.01        52.40          459
                   t        78.37        44.65        56.89          430

               micro        70.74        44.32        54.50          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        51.06        37.97        43.56          316
                 NEG        40.00         1.41         2.72          142
                 NEU         0.00         0.00         0.00           25

               micro        50.83        25.26        33.75          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        51.06        37.97        43.56          316
                 NEG        40.00         1.41         2.72          142
                 NEU         0.00         0.00         0.00           25

               micro        50.83        25.26        33.75          483

Train epoch 16: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 17: 100%|████████████████████████| 27/27 [00:03<00:00,  7.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.25        65.36        53.48          459
                   t        56.51        62.56        59.38          430

               micro        49.96        64.00        56.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        24.96        49.68        33.23          316
                 NEG        29.17         9.86        14.74          142
                 NEU         0.00         0.00         0.00           25

               micro        25.26        35.40        29.48          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        24.96        49.68        33.23          316
                 NEG        29.17         9.86        14.74          142
                 NEU         0.00         0.00         0.00           25

               micro        25.26        35.40        29.48          483

Train epoch 17: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 18: 100%|████████████████████████| 27/27 [00:03<00:00,  8.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        61.69        55.77        58.58          459
                   t        83.33        44.19        57.75          430

               micro        69.36        50.17        58.22          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        56.60        37.97        45.45          316
                 NEG        20.75         7.75        11.28          142
                 NEU         0.00         0.00         0.00           25

               micro        49.43        27.12        35.03          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        56.60        37.97        45.45          316
                 NEG        20.75         7.75        11.28          142
                 NEU         0.00         0.00         0.00           25

               micro        49.43        27.12        35.03          483

Train epoch 18: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 19: 100%|████████████████████████| 27/27 [00:03<00:00,  8.07it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.85        50.98        58.94          459
                   t        67.06        53.02        59.22          430

               micro        68.44        51.97        59.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        44.57        37.66        40.82          316
                 NEG        17.65         2.11         3.77          142
                 NEU         0.00         0.00         0.00           25

               micro        42.96        25.26        31.81          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        44.57        37.66        40.82          316
                 NEG        17.65         2.11         3.77          142
                 NEU         0.00         0.00         0.00           25

               micro        42.96        25.26        31.81          483

Train epoch 19: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 20: 100%|████████████████████████| 27/27 [00:03<00:00,  7.93it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        63.76        59.04        61.31          459
                   t        59.86        60.00        59.93          430

               micro        61.80        59.51        60.63          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        34.17        43.04        38.10          316
                 NEG        22.94        17.61        19.92          142
                 NEU         0.00         0.00         0.00           25

               micro        31.69        33.33        32.49          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        33.67        42.41        37.54          316
                 NEG        22.94        17.61        19.92          142
                 NEU         0.00         0.00         0.00           25

               micro        31.30        32.92        32.09          483

Train epoch 20: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 21: 100%|████████████████████████| 27/27 [00:03<00:00,  8.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        62.47        59.48        60.94          459
                   t        81.33        42.56        55.88          430

               micro        68.88        51.29        58.80          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        57.21        38.92        46.33          316
                 NEG        23.08        10.56        14.49          142
                 NEU         0.00         0.00         0.00           25

               micro        49.29        28.57        36.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        56.74        38.61        45.95          316
                 NEG        23.08        10.56        14.49          142
                 NEU         0.00         0.00         0.00           25

               micro        48.93        28.36        35.91          483

Train epoch 21: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 22: 100%|████████████████████████| 27/27 [00:03<00:00,  8.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        37.84         3.05         5.65          459
                   t        38.46         5.81        10.10          430

               micro        38.24         4.39         7.87          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        12.50         0.63         1.20          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro         7.41         0.41         0.78          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        12.50         0.63         1.20          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro         7.41         0.41         0.78          483

Train epoch 22: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 23: 100%|████████████████████████| 27/27 [00:03<00:00,  8.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 23: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 24: 100%|████████████████████████| 27/27 [00:03<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 24: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 25: 100%|████████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 25: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 26: 100%|████████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 26: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 27: 100%|████████████████████████| 27/27 [00:03<00:00,  8.62it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 27: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 28: 100%|████████████████████████| 27/27 [00:03<00:00,  8.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 28: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 29: 100%|████████████████████████| 27/27 [00:03<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        20.00         0.93         1.78          430

               micro        20.00         0.45         0.88          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 29: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 30: 100%|████████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.11         2.83         5.25          459
                   t        30.00         2.09         3.91          430

               micro        33.33         2.47         4.61          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        15.79         1.90         3.39          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        15.79         1.24         2.30          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 30: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 31: 100%|████████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.11         2.83         5.25          459
                   t        27.27         2.09         3.89          430

               micro        31.88         2.47         4.59          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        17.86         1.58         2.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        17.86         1.04         1.96          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 31: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 32: 100%|████████████████████████| 27/27 [00:03<00:00,  8.62it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00        459.0
                   t         0.00         0.00         0.00        430.0

               micro         0.00         0.00         0.00        889.0


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 32: 100%|███████████████████████████| 49/49 [00:16<00:00,  3.00it/s]
Evaluate epoch 33: 100%|████████████████████████| 27/27 [00:03<00:00,  8.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.00         4.79         8.64          459
                   t        26.83         2.56         4.67          430

               micro        36.26         3.71         6.73          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        18.75         1.90         3.45          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        18.75         1.24         2.33          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 33: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 34: 100%|████████████████████████| 27/27 [00:03<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.36         6.10        10.53          459
                   t        30.00         2.09         3.91          430

               micro        35.92         4.16         7.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        20.00         1.58         2.93          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        20.00         1.04         1.97          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 34: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 35: 100%|████████████████████████| 27/27 [00:03<00:00,  8.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        23.81         1.09         2.08          459
                   t        29.41         1.16         2.24          430

               micro        26.32         1.12         2.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        41.67         1.58         3.05          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        41.67         1.04         2.02          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 35: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 36: 100%|████████████████████████| 27/27 [00:03<00:00,  8.66it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.76         5.45         9.49          459
                   t        50.00         1.16         2.27          430

               micro        38.46         3.37         6.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 36: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 37: 100%|████████████████████████| 27/27 [00:03<00:00,  8.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        50.00         1.16         2.27          430

               micro        50.00         0.56         1.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 37: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 38: 100%|████████████████████████| 27/27 [00:03<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.00         0.65         1.29          459
                   t         0.00         0.00         0.00          430

               micro        37.50         0.34         0.67          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 38: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 39: 100%|████████████████████████| 27/27 [00:03<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        14.29         0.44         0.85          459
                   t        50.00         1.16         2.27          430

               micro        29.17         0.79         1.53          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 39: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 40: 100%|████████████████████████| 27/27 [00:03<00:00,  8.44it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        31.39         9.37        14.43          459
                   t         5.00         0.23         0.44          430

               micro        28.03         4.95         8.41          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 40: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 41: 100%|████████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.00         2.40         4.57          459
                   t        33.33         1.86         3.52          430

               micro        41.30         2.14         4.06          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 41: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 42: 100%|████████████████████████| 27/27 [00:03<00:00,  8.66it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        36.76         5.45         9.49          459
                   t        50.00         1.16         2.27          430

               micro        38.46         3.37         6.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 42: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 43: 100%|████████████████████████| 27/27 [00:03<00:00,  8.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.50         3.70         6.81          459
                   t        83.33         1.16         2.29          430

               micro        47.83         2.47         4.71          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 43: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 44: 100%|████████████████████████| 27/27 [00:03<00:00,  8.68it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        33.33         1.31         2.52          459
                   t        83.33         1.16         2.29          430

               micro        45.83         1.24         2.41          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 44: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 45: 100%|████████████████████████| 27/27 [00:03<00:00,  8.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.50         3.70         6.81          459
                   t        83.33         1.16         2.29          430

               micro        47.83         2.47         4.71          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 45: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 46: 100%|████████████████████████| 27/27 [00:03<00:00,  8.93it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.50         3.70         6.81          459
                   t        83.33         1.16         2.29          430

               micro        47.83         2.47         4.71          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 46: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.99it/s]
Evaluate epoch 47: 100%|████████████████████████| 27/27 [00:03<00:00,  8.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        39.29         2.40         4.52          459
                   t        83.33         1.16         2.29          430

               micro        47.06         1.80         3.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 47: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 48: 100%|████████████████████████| 27/27 [00:03<00:00,  8.73it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.06         5.23         9.49          459
                   t        49.15         6.74        11.86          430

               micro        50.00         5.96        10.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        48.15         4.11         7.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        48.15         2.69         5.10          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        48.15         4.11         7.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        48.15         2.69         5.10          483

Train epoch 48: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 49: 100%|████████████████████████| 27/27 [00:03<00:00,  8.75it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.28         6.32        10.92          459
                   t        45.71         7.44        12.80          430

               micro        42.96         6.86        11.83          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        38.10         2.53         4.75          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.10         1.66         3.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        38.10         2.53         4.75          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.10         1.66         3.17          483

Train epoch 49: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 50: 100%|████████████████████████| 27/27 [00:03<00:00,  8.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.69        12.85        19.54          459
                   t        83.33         1.16         2.29          430

               micro        42.38         7.20        12.31          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 50: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.99it/s]
Evaluate epoch 51: 100%|████████████████████████| 27/27 [00:03<00:00,  8.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         2.40         4.68          459
                   t        47.22         7.91        13.55          430

               micro        54.22         5.06         9.26          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 51: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 52: 100%|████████████████████████| 27/27 [00:03<00:00,  8.84it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.06         5.23         9.49          459
                   t        83.33         1.16         2.29          430

               micro        54.72         3.26         6.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 52: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 53: 100%|████████████████████████| 27/27 [00:03<00:00,  8.37it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.00         6.10        11.00          459
                   t        50.79         7.44        12.98          430

               micro        53.10         6.75        11.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        40.43         6.01        10.47          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        40.43         3.93         7.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        40.43         6.01        10.47          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        40.43         3.93         7.17          483

Train epoch 53: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 54: 100%|████████████████████████| 27/27 [00:03<00:00,  8.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.06         5.23         9.49          459
                   t        83.33         1.16         2.29          430

               micro        54.72         3.26         6.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 54: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 55: 100%|████████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.57         7.84        13.38          459
                   t        55.56         5.81        10.53          430

               micro        49.19         6.86        12.04          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        37.21         5.06         8.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        37.21         3.31         6.08          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        37.21         5.06         8.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        37.21         3.31         6.08          483

Train epoch 55: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 56: 100%|████████████████████████| 27/27 [00:03<00:00,  8.38it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        41.71        15.90        23.03          459
                   t        52.38         7.67        13.39          430

               micro        44.54        11.92        18.81          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        33.33         6.33        10.64          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        33.33         4.14         7.37          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        31.67         6.01        10.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        31.67         3.93         7.00          483

Train epoch 56: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 57: 100%|████████████████████████| 27/27 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         2.40         4.68          459
                   t        62.50         4.65         8.66          430

               micro        72.09         3.49         6.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        80.00         2.53         4.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        80.00         1.66         3.25          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        80.00         2.53         4.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        80.00         1.66         3.25          483

Train epoch 57: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 58: 100%|████████████████████████| 27/27 [00:03<00:00,  8.28it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        55.56         5.81        10.53          430

               micro        51.09         7.87        13.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 58: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 59: 100%|████████████████████████| 27/27 [00:03<00:00,  8.34it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        64.29         1.96         3.81          459
                   t        83.33         1.16         2.29          430

               micro        70.00         1.57         3.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 59: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 60: 100%|████████████████████████| 27/27 [00:03<00:00,  8.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        66.67         4.79         8.94          459
                   t        83.33         1.16         2.29          430

               micro        69.23         3.04         5.82          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 60: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 61: 100%|████████████████████████| 27/27 [00:03<00:00,  8.48it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        64.29         1.96         3.81          459
                   t        83.33         1.16         2.29          430

               micro        70.00         1.57         3.08          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 61: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 62: 100%|████████████████████████| 27/27 [00:03<00:00,  8.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.88         6.54        11.47          459
                   t        55.56         5.81        10.53          430

               micro        50.46         6.19        11.02          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        37.21         5.06         8.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        37.21         3.31         6.08          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        37.21         5.06         8.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        37.21         3.31         6.08          483

Train epoch 62: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 63: 100%|████████████████████████| 27/27 [00:03<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.53         5.45        10.14          459
                   t        83.33         1.16         2.29          430

               micro        75.00         3.37         6.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 63: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 64: 100%|████████████████████████| 27/27 [00:03<00:00,  8.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.33         5.66        10.02          459
                   t        50.00         1.16         2.27          430

               micro        44.29         3.49         6.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 64: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 65: 100%|████████████████████████| 27/27 [00:03<00:00,  8.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        67.39         6.75        12.28          459
                   t        83.33         1.16         2.29          430

               micro        69.23         4.05         7.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 65: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 66: 100%|████████████████████████| 27/27 [00:03<00:00,  8.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        55.56         5.81        10.53          430

               micro        52.46         7.20        12.66          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        30.36         5.38         9.14          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        30.36         3.52         6.31          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         5.06         8.60          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         3.31         5.94          483

Train epoch 66: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 67: 100%|████████████████████████| 27/27 [00:03<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.36        12.85        19.93          459
                   t        55.56         5.81        10.53          430

               micro        47.19         9.45        15.75          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.67         3.80         6.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.67         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        24.44         3.48         6.09          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.44         2.28         4.17          483

Train epoch 67: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 68: 100%|████████████████████████| 27/27 [00:03<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.88         6.54        11.47          459
                   t        39.76         7.67        12.87          430

               micro        42.86         7.09        12.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        30.36         5.38         9.14          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        30.36         3.52         6.31          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        30.36         5.38         9.14          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        30.36         3.52         6.31          483

Train epoch 68: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 69: 100%|████████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        62.50         2.33         4.48          430

               micro        50.93         6.19        11.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 69: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 70: 100%|████████████████████████| 27/27 [00:03<00:00,  8.18it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        62.50         2.33         4.48          430

               micro        50.93         6.19        11.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 70: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 71: 100%|████████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.39         8.06        13.94          459
                   t        55.56         5.81        10.53          430

               micro        52.99         6.97        12.33          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        38.46         4.75         8.45          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.46         3.11         5.75          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        38.46         4.75         8.45          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.46         3.11         5.75          483

Train epoch 71: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 72: 100%|████████████████████████| 27/27 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        62.50         2.33         4.48          430

               micro        52.69         5.51         9.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 72: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 73: 100%|████████████████████████| 27/27 [00:03<00:00,  8.48it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.57         7.84        13.38          459
                   t        62.50         2.33         4.48          430

               micro        48.42         5.17         9.35          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 73: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 74: 100%|████████████████████████| 27/27 [00:03<00:00,  8.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        57.58         4.42         8.21          430

               micro        51.20         7.20        12.62          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483

Train epoch 74: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 75: 100%|████████████████████████| 27/27 [00:03<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.19         3.49         6.67          459
                   t        83.33         1.16         2.29          430

               micro        77.78         2.36         4.59          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 75: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 76: 100%|████████████████████████| 27/27 [00:03<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        62.50         2.33         4.48          430

               micro        52.69         5.51         9.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 76: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 77: 100%|████████████████████████| 27/27 [00:03<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        62.50         2.33         4.48          430

               micro        50.93         6.19        11.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 77: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 78: 100%|████████████████████████| 27/27 [00:03<00:00,  8.36it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.25         5.88        10.65          459
                   t        83.33         1.16         2.29          430

               micro        59.26         3.60         6.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        31.58         1.90         3.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        31.58         1.24         2.39          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        26.32         1.58         2.99          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483

Train epoch 78: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 79: 100%|████████████████████████| 27/27 [00:03<00:00,  8.17it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        57.58         4.42         8.21          430

               micro        51.20         7.20        12.62          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483

Train epoch 79: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 80: 100%|████████████████████████| 27/27 [00:03<00:00,  8.30it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        62.50         2.33         4.48          430

               micro        52.69         5.51         9.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 80: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.72it/s]
Evaluate epoch 81: 100%|████████████████████████| 27/27 [00:03<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.65         8.50        14.55          459
                   t        62.50         2.33         4.48          430

               micro        52.69         5.51         9.98          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 81: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 82: 100%|████████████████████████| 27/27 [00:03<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.59         8.93        14.99          459
                   t        46.03         6.74        11.76          430

               micro        46.36         7.87        13.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        21.05         3.80         6.43          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        17.39         2.48         4.35          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.30         3.48         5.90          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        15.94         2.28         3.99          483

Train epoch 82: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 83: 100%|████████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.57         7.84        13.38          459
                   t        83.33         1.16         2.29          430

               micro        48.24         4.61         8.42          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 83: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 84: 100%|████████████████████████| 27/27 [00:03<00:00,  8.72it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        48.94         5.35         9.64          430

               micro        48.92         7.65        13.23          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        21.05         3.80         6.43          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        21.05         2.48         4.44          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.30         3.48         5.90          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.30         2.28         4.07          483

Train epoch 84: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 85: 100%|████████████████████████| 27/27 [00:03<00:00,  8.48it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.59         8.93        14.99          459
                   t        51.02         5.81        10.44          430

               micro        48.18         7.42        12.87          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483

Train epoch 85: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 86: 100%|████████████████████████| 27/27 [00:03<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.57         7.84        13.38          459
                   t        43.48         4.65         8.40          430

               micro        44.80         6.30        11.05          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483

Train epoch 86: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 87: 100%|████████████████████████| 27/27 [00:03<00:00,  8.62it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        57.89         4.79         8.85          459
                   t        57.58         4.42         8.21          430

               micro        57.75         4.61         8.54          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 87: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 88: 100%|████████████████████████| 27/27 [00:03<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.57         7.84        13.38          459
                   t        43.48         4.65         8.40          430

               micro        44.80         6.30        11.05          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483

Train epoch 88: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 89: 100%|████████████████████████| 27/27 [00:03<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        55.56         5.81        10.53          430

               micro        51.09         7.87        13.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        36.36         2.53         4.73          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        36.36         1.66         3.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        36.36         2.53         4.73          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        36.36         1.66         3.17          483

Train epoch 89: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 90: 100%|████████████████████████| 27/27 [00:03<00:00,  8.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.25         5.88        10.65          459
                   t        83.33         1.16         2.29          430

               micro        59.26         3.60         6.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 90: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 91: 100%|████████████████████████| 27/27 [00:03<00:00,  8.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.75         8.50        14.47          459
                   t        57.58         4.42         8.21          430

               micro        51.33         6.52        11.58          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        35.48         3.48         6.34          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        35.48         2.28         4.28          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        35.48         3.48         6.34          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        35.48         2.28         4.28          483

Train epoch 91: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 92: 100%|████████████████████████| 27/27 [00:03<00:00,  8.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.00         8.50        14.61          459
                   t        62.50         2.33         4.48          430

               micro        53.85         5.51        10.00          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 92: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 93: 100%|████████████████████████| 27/27 [00:03<00:00,  8.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        42.67         6.97        11.99          459
                   t        34.38         5.12         8.91          430

               micro        38.85         6.07        10.51          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         2.48         4.55          483

Train epoch 93: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 94: 100%|████████████████████████| 27/27 [00:03<00:00,  8.66it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.59         8.93        14.99          459
                   t        45.10         5.35         9.56          430

               micro        46.04         7.20        12.45          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483

Train epoch 94: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 95: 100%|████████████████████████| 27/27 [00:03<00:00,  8.68it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.53         5.45        10.14          459
                   t        62.50         2.33         4.48          430

               micro        70.00         3.94         7.45          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 95: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 96: 100%|████████████████████████| 27/27 [00:03<00:00,  8.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        62.50         2.33         4.48          430

               micro        50.93         6.19        11.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 96: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 97: 100%|████████████████████████| 27/27 [00:03<00:00,  8.77it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.51         4.36         7.97          459
                   t        40.00         4.65         8.33          430

               micro        43.01         4.50         8.15          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483

Train epoch 97: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 98: 100%|████████████████████████| 27/27 [00:03<00:00,  8.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.25         5.88        10.65          459
                   t        83.33         1.16         2.29          430

               micro        59.26         3.60         6.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483

Train epoch 98: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 99: 100%|████████████████████████| 27/27 [00:03<00:00,  8.45it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        44.83         6.05        10.66          430

               micro        47.33         7.99        13.67          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         2.28         4.17          483

Train epoch 99: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 100: 100%|███████████████████████| 27/27 [00:03<00:00,  8.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        44.83         5.66        10.06          459
                   t        40.00         4.65         8.33          430

               micro        42.59         5.17         9.23          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483

Train epoch 100: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 101: 100%|███████████████████████| 27/27 [00:03<00:00,  8.57it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.94        10.02        16.64          459
                   t        48.94         5.35         9.64          430

               micro        48.94         7.76        13.40          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        20.69         3.80         6.42          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        20.69         2.48         4.44          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        18.97         3.48         5.88          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        18.97         2.28         4.07          483

Train epoch 101: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 102: 100%|███████████████████████| 27/27 [00:03<00:00,  8.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         0.87         1.73          459
                   t        83.33         1.16         2.29          430

               micro        90.00         1.01         2.00          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 102: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 103: 100%|███████████████████████| 27/27 [00:03<00:00,  8.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.19         3.49         6.67          459
                   t        43.48         4.65         8.40          430

               micro        53.73         4.05         7.53          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 103: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 104: 100%|███████████████████████| 27/27 [00:03<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        42.86         5.58         9.88          430

               micro        46.62         7.76        13.31          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.30         3.48         5.90          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.30         2.28         4.07          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.30         3.48         5.90          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.30         2.28         4.07          483

Train epoch 104: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 105: 100%|███████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.00         8.50        14.61          459
                   t        57.58         4.42         8.21          430

               micro        53.70         6.52        11.63          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 105: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 106: 100%|███████████████████████| 27/27 [00:03<00:00,  8.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.91         9.80        16.33          459
                   t        55.88         4.42         8.19          430

               micro        50.79         7.20        12.61          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.27         3.80         6.67          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        20.00         2.48         4.42          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         3.48         6.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        18.33         2.28         4.05          483

Train epoch 106: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 107: 100%|███████████████████████| 27/27 [00:03<00:00,  8.82it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.38         7.19        12.64          459
                   t        57.58         4.42         8.21          430

               micro        54.17         5.85        10.56          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 107: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 108: 100%|███████████████████████| 27/27 [00:03<00:00,  8.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.32         8.50        14.58          459
                   t        44.23         5.35         9.54          430

               micro        48.44         6.97        12.19          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 108: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 109: 100%|███████████████████████| 27/27 [00:03<00:00,  8.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.46        10.02        16.67          459
                   t        46.94         5.35         9.60          430

               micro        48.59         7.76        13.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        24.44         3.48         6.09          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.44         2.28         4.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        24.44         3.48         6.09          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.44         2.28         4.17          483

Train epoch 109: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 110: 100%|███████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.46        10.02        16.49          459
                   t        57.58         4.42         8.21          430

               micro        49.24         7.31        12.73          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        24.00         3.80         6.56          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        24.00         2.48         4.50          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        22.00         3.48         6.01          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        22.00         2.28         4.13          483

Train epoch 110: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 111: 100%|███████████████████████| 27/27 [00:03<00:00,  8.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.32         8.50        14.58          459
                   t        50.00         4.42         8.12          430

               micro        50.88         6.52        11.57          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 111: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 112: 100%|███████████████████████| 27/27 [00:03<00:00,  8.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.79         9.80        15.65          459
                   t        42.59         5.35         9.50          430

               micro        40.00         7.65        12.84          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.64         3.48         5.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        14.86         2.28         3.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.64         3.48         5.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        14.86         2.28         3.95          483

Train epoch 112: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.99it/s]
Evaluate epoch 113: 100%|███████████████████████| 27/27 [00:03<00:00,  8.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.38         7.19        12.64          459
                   t        62.50         2.33         4.48          430

               micro        54.43         4.84         8.88          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.57         2.53         4.65          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.57         1.66         3.13          483

Train epoch 113: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 114: 100%|███████████████████████| 27/27 [00:03<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.38         7.19        12.64          459
                   t        48.94         5.35         9.64          430

               micro        50.91         6.30        11.21          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        16.67         2.85         4.86          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        16.67         1.86         3.35          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        14.81         2.53         4.32          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        14.81         1.66         2.98          483

Train epoch 114: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 115: 100%|███████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.00         8.50        14.61          459
                   t        62.50         2.33         4.48          430

               micro        53.85         5.51        10.00          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 115: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 116: 100%|███████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        70.00         4.58         8.59          459
                   t        29.63         1.86         3.50          430

               micro        50.88         3.26         6.13          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.32         1.58         2.99          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        26.32         1.58         2.99          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483

Train epoch 116: 100%|██████████████████████████| 49/49 [00:16<00:00,  3.00it/s]
Evaluate epoch 117: 100%|███████████████████████| 27/27 [00:03<00:00,  8.46it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        52.00         8.50        14.61          459
                   t        57.58         4.42         8.21          430

               micro        53.70         6.52        11.63          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 117: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 118: 100%|███████████████████████| 27/27 [00:03<00:00,  8.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        61.29         8.28        14.59          459
                   t        38.24         6.05        10.44          430

               micro        49.23         7.20        12.56          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         2.53         4.60          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         1.86         3.49          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         2.53         4.60          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        27.27         1.86         3.49          483

Train epoch 118: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 119: 100%|███████████████████████| 27/27 [00:03<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.39         6.54        11.52          459
                   t        37.93         2.56         4.79          430

               micro        45.05         4.61         8.37          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483

Train epoch 119: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 120: 100%|███████████████████████| 27/27 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        37.50         5.23         9.18          459
                   t        35.82         5.58         9.66          430

               micro        36.64         5.40         9.41          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        11.69         2.85         4.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.69         1.86         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        11.69         2.85         4.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.69         1.86         3.21          483

Train epoch 120: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 121: 100%|███████████████████████| 27/27 [00:03<00:00,  8.50it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.00         8.71        14.84          459
                   t        52.38         2.56         4.88          430

               micro        50.50         5.74        10.30          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        28.12         2.85         5.17          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.12         1.86         3.50          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         2.53         4.60          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         1.66         3.11          483

Train epoch 121: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 122: 100%|███████████████████████| 27/27 [00:03<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.15         6.32        11.20          459
                   t        57.58         4.42         8.21          430

               micro        52.17         5.40         9.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        53.33         1.66         3.21          483

Train epoch 122: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 123: 100%|███████████████████████| 27/27 [00:03<00:00,  8.32it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.00        10.89        17.12          459
                   t        34.67         6.05        10.30          430

               micro        38.00         8.55        13.96          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.64         3.48         5.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.64         2.28         4.08          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.64         3.48         5.91          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.64         2.28         4.08          483

Train epoch 123: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.88it/s]
Evaluate epoch 124: 100%|███████████████████████| 27/27 [00:03<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.88         6.54        11.47          459
                   t        52.63         2.33         4.45          430

               micro        48.19         4.50         8.23          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483

Train epoch 124: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 125: 100%|███████████████████████| 27/27 [00:03<00:00,  8.35it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        33.90         4.36         7.72          459
                   t        43.24         7.44        12.70          430

               micro        39.10         5.85        10.18          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        11.11         1.27         2.27          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.11         0.83         1.54          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        11.11         1.27         2.27          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.11         0.83         1.54          483

Train epoch 125: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 126: 100%|███████████████████████| 27/27 [00:03<00:00,  8.44it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        51.95         8.71        14.93          459
                   t        85.71         1.40         2.75          430

               micro        54.76         5.17         9.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        83.33         1.58         3.11          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        83.33         1.04         2.04          483

Train epoch 126: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 127: 100%|███████████████████████| 27/27 [00:03<00:00,  8.53it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        55.10         5.88        10.63          459
                   t        57.89         2.56         4.90          430

               micro        55.88         4.27         7.94          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483

Train epoch 127: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 128: 100%|███████████████████████| 27/27 [00:03<00:00,  8.48it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        34.12         6.32        10.66          459
                   t        39.68         5.81        10.14          430

               micro        36.49         6.07        10.41          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        21.43         2.85         5.03          316
                 NEG         7.14         0.70         1.28          142
                 NEU         0.00         0.00         0.00           25

               micro        17.86         2.07         3.71          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        21.43         2.85         5.03          316
                 NEG         7.14         0.70         1.28          142
                 NEU         0.00         0.00         0.00           25

               micro        17.86         2.07         3.71          483

Train epoch 128: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 129: 100%|███████████████████████| 27/27 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        47.92        10.02        16.58          459
                   t        44.07         6.05        10.63          430

               micro        46.45         8.10        13.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        20.69         3.80         6.42          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        22.03         2.69         4.80          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        18.97         3.48         5.88          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        20.34         2.48         4.43          483

Train epoch 129: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 130: 100%|███████████████████████| 27/27 [00:03<00:00,  8.42it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.39         6.54        11.52          459
                   t        24.49         2.79         5.01          430

               micro        37.84         4.72         8.40          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        13.04         2.85         4.68          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        13.04         1.86         3.26          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        13.04         2.85         4.68          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        13.04         1.86         3.26          483

Train epoch 130: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 131: 100%|███████████████████████| 27/27 [00:03<00:00,  8.47it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        41.25         7.19        12.24          459
                   t        47.06         5.58         9.98          430

               micro        43.51         6.41        11.18          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.51         2.53         4.48          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.05         1.66         3.05          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.51         2.53         4.48          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.05         1.66         3.05          483

Train epoch 131: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 132: 100%|███████████████████████| 27/27 [00:03<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        35.33        11.55        17.41          459
                   t        57.14         2.79         5.32          430

               micro        38.01         7.31        12.26          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        42.11         1.66         3.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        42.11         1.66         3.19          483

Train epoch 132: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 133: 100%|███████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.98         6.75        11.21          459
                   t        40.00         2.79         5.22          430

               micro        34.68         4.84         8.49          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        32.14         2.85         5.23          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        32.14         1.86         3.52          483

Train epoch 133: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 134: 100%|███████████████████████| 27/27 [00:03<00:00,  8.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        50.00         6.54        11.56          459
                   t        51.02         5.81        10.44          430

               micro        50.46         6.19        11.02          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        27.59         2.53         4.64          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        30.00         1.86         3.51          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.59         2.53         4.64          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        30.00         1.86         3.51          483

Train epoch 134: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.98it/s]
Evaluate epoch 135: 100%|███████████████████████| 27/27 [00:03<00:00,  8.51it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.53        11.76        17.28          459
                   t        33.80         5.58         9.58          430

               micro        32.91         8.77        13.85          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        21.95         2.85         5.04          316
                 NEG        50.00         0.70         1.39          142
                 NEU         0.00         0.00         0.00           25

               micro        23.26         2.07         3.80          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.51         2.53         4.48          316
                 NEG        50.00         0.70         1.39          142
                 NEU         0.00         0.00         0.00           25

               micro        20.93         1.86         3.42          483

Train epoch 135: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 136: 100%|███████████████████████| 27/27 [00:03<00:00,  8.54it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.25         5.88        10.65          459
                   t        43.86         5.81        10.27          430

               micro        49.52         5.85        10.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.67         2.53         4.62          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        17.78         1.66         3.03          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        26.67         2.53         4.62          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        17.78         1.66         3.03          483

Train epoch 136: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 137: 100%|███████████████████████| 27/27 [00:03<00:00,  8.43it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        33.33         5.23         9.04          459
                   t        28.26         9.07        13.73          430

               micro        30.00         7.09        11.46          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        16.36         2.85         4.85          316
                 NEG         6.67         0.70         1.27          142
                 NEU         0.00         0.00         0.00           25

               micro        14.29         2.07         3.62          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        14.55         2.53         4.31          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        11.43         1.66         2.89          483

Train epoch 137: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.96it/s]
Evaluate epoch 138: 100%|███████████████████████| 27/27 [00:03<00:00,  8.49it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.01         8.71        14.49          459
                   t        52.00         3.02         5.71          430

               micro        44.92         5.96        10.53          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        29.03         2.85         5.19          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        29.03         1.86         3.50          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.81         2.53         4.61          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.81         1.66         3.11          483

Train epoch 138: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 139: 100%|███████████████████████| 27/27 [00:03<00:00,  8.31it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.21         6.75        11.88          459
                   t        50.00         2.56         4.87          430

               micro        49.41         4.72         8.62          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        47.06         2.53         4.80          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        47.06         1.66         3.20          483

Train epoch 139: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 140: 100%|███████████████████████| 27/27 [00:03<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.67         6.10        10.79          459
                   t        40.00         1.40         2.70          430

               micro        45.33         3.82         7.05          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        55.56         1.58         3.08          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        55.56         1.04         2.03          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        55.56         1.58         3.08          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        55.56         1.04         2.03          483

Train epoch 140: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.97it/s]
Evaluate epoch 141: 100%|███████████████████████| 27/27 [00:03<00:00,  8.40it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        34.41         6.97        11.59          459
                   t        37.21         3.72         6.77          430

               micro        35.29         5.40         9.37          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        20.45         2.85         5.00          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        20.45         1.86         3.42          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        18.18         2.53         4.44          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        18.18         1.66         3.04          483

Train epoch 141: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 142: 100%|███████████████████████| 27/27 [00:03<00:00,  8.68it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        56.36         6.75        12.06          459
                   t        46.43         3.02         5.68          430

               micro        53.01         4.95         9.05          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.10         1.66         3.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.33         2.53         4.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.10         1.66         3.17          483

Train epoch 142: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 143: 100%|███████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.82         7.19        12.13          459
                   t        31.11         6.51        10.77          430

               micro        34.86         6.86        11.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.35         3.80         6.35          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.35         2.48         4.40          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        19.35         3.80         6.35          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.35         2.48         4.40          483

Train epoch 143: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 144: 100%|███████████████████████| 27/27 [00:03<00:00,  8.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.90         7.84        13.31          459
                   t        87.50         1.63         3.20          430

               micro        47.78         4.84         8.78          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.32         1.58         2.99          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        26.32         1.58         2.99          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.32         1.04         1.99          483

Train epoch 144: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 145: 100%|███████████████████████| 27/27 [00:03<00:00,  8.52it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.65         6.97        11.49          459
                   t        31.11         3.26         5.89          430

               micro        32.17         5.17         8.91          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        23.08         2.85         5.07          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        23.08         1.86         3.45          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        23.08         2.85         5.07          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        23.08         1.86         3.45          483

Train epoch 145: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.95it/s]
Evaluate epoch 146: 100%|███████████████████████| 27/27 [00:03<00:00,  8.62it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.30         5.88        10.27          459
                   t        43.08         6.51        11.31          430

               micro        41.67         6.19        10.77          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        31.82         4.43         7.78          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        33.33         3.11         5.68          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        31.82         4.43         7.78          316
                 NEG       100.00         0.70         1.40          142
                 NEU         0.00         0.00         0.00           25

               micro        33.33         3.11         5.68          483

Train epoch 146: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 147: 100%|███████████████████████| 27/27 [00:03<00:00,  8.58it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        48.78         8.71        14.79          459
                   t        55.26         4.88         8.97          430

               micro        50.83         6.86        12.09          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        44.44         2.53         4.79          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        44.44         1.66         3.19          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        44.44         2.53         4.79          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        44.44         1.66         3.19          483

Train epoch 147: 100%|██████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 148: 100%|███████████████████████| 27/27 [00:03<00:00,  8.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        35.40         8.71        13.99          459
                   t        54.55         2.79         5.31          430

               micro        38.52         5.85        10.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         0.95         1.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         0.62         1.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         0.95         1.83          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         0.62         1.21          483

Train epoch 148: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.93it/s]
Evaluate epoch 149: 100%|███████████████████████| 27/27 [00:03<00:00,  8.71it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.30         7.84        13.02          459
                   t        34.52         6.74        11.28          430

               micro        36.52         7.31        12.18          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        25.00         2.53         4.60          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         1.66         3.11          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        25.00         2.53         4.60          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        25.00         1.66         3.11          483

Train epoch 149: 100%|██████████████████████████| 49/49 [00:16<00:00,  2.94it/s]
Evaluate epoch 150: 100%|███████████████████████| 27/27 [00:03<00:00,  8.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        39.60         8.71        14.29          459
                   t        33.33         3.26         5.93          430

               micro        37.76         6.07        10.47          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        16.07         2.85         4.84          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        15.00         1.86         3.31          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        14.29         2.53         4.30          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        13.33         1.66         2.95          483


Best F1 score: 40 at epoch -1