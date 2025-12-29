🚀 Starting training with parameters:
  - Model: microsoft/deberta-v3-base
  - emb_dim: 768
  - hidden_dim: 384 (bidirectional → 768)
  - deberta_feature_dim: 768
  - gcn_dim: 768
  - mem_dim: 768
============================================================
2025-12-29 08:05:52.999391: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:477] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered
WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
E0000 00:00:1766995553.019458    8816 cuda_dnn.cc:8310] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered
E0000 00:00:1766995553.025451    8816 cuda_blas.cc:1418] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered
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
Parse dataset 'train': 100%|████████████████| 592/592 [00:00<00:00, 1589.24it/s]
Parse dataset 'test': 100%|█████████████████| 320/320 [00:00<00:00, 1514.81it/s]
    15res    8
Some weights of the model checkpoint at microsoft/deberta-v3-base were not used when initializing D2E2SModel: ['lm_predictions.lm_head.dense.weight', 'mask_predictions.dense.bias', 'mask_predictions.dense.weight', 'lm_predictions.lm_head.LayerNorm.bias', 'deberta.embeddings.position_embeddings.weight', 'mask_predictions.classifier.weight', 'mask_predictions.LayerNorm.bias', 'lm_predictions.lm_head.LayerNorm.weight', 'mask_predictions.classifier.bias', 'lm_predictions.lm_head.bias', 'mask_predictions.LayerNorm.weight', 'lm_predictions.lm_head.dense.bias']
- This IS expected if you are initializing D2E2SModel from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing D2E2SModel from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of D2E2SModel were not initialized from the model checkpoint at microsoft/deberta-v3-base and are newly initialized: ['Sem_gcn.W.0.bias', 'lstm.bias_ih_l1_reverse', 'TIN.GatedGCN.conv1.lin_l.bias', 'TIN.residual_layer1.3.weight', 'entity_classifier.weight', 'TIN.GatedGCN.conv3.lin_r.bias', 'TIN.residual_layer3.2.bias', 'TIN.lstm.weight_ih_l0', 'TIN.lstm.weight_hh_l1', 'TIN.GatedGCN.conv1.bias', 'TIN.GatedGCN.conv2.lin_l.bias', 'TIN.residual_layer1.0.weight', 'lstm.weight_hh_l1', 'lstm.weight_ih_l1', 'senti_classifier.bias', 'TIN.GatedGCN.conv3.att', 'Sem_gcn.attn.linears.1.weight', 'TIN.residual_layer3.3.weight', 'TIN.feature_fusion.2.bias', 'TIN.lstm.bias_hh_l0', 'TIN.GatedGCN.conv2.lin_l.weight', 'TIN.lstm.bias_ih_l1_reverse', 'TIN.GatedGCN.conv3.lin_l.bias', 'TIN.feature_fusion.2.weight', 'TIN.lstm.weight_ih_l1', 'TIN.GatedGCN.conv2.att', 'deberta.embeddings.position_ids', 'Sem_gcn.attn.linears.0.weight', 'TIN.residual_layer3.0.weight', 'lstm.bias_ih_l0_reverse', 'TIN.GatedGCN.conv3.bias', 'entity_classifier.bias', 'TIN.lstm.weight_hh_l1_reverse', 'lstm.bias_hh_l0', 'Syn_gcn.W.0.weight', 'TIN.residual_layer4.3.weight', 'TIN.residual_layer3.2.weight', 'fc.bias', 'TIN.residual_layer1.2.weight', 'Sem_gcn.attn.linears.0.bias', 'lstm.bias_ih_l1', 'TIN.lstm.weight_hh_l0_reverse', 'Sem_gcn.W.1.weight', 'TIN.GatedGCN.conv1.att', 'TIN.GatedGCN.conv1.lin_r.weight', 'TIN.residual_layer3.3.bias', 'attention_layer.w_value.weight', 'Syn_gcn.W.1.weight', 'TIN.residual_layer1.3.bias', 'TIN.residual_layer2.3.weight', 'TIN.GatedGCN.conv2.lin_r.weight', 'lstm.weight_hh_l1_reverse', 'TIN.lstm.bias_ih_l0_reverse', 'TIN.residual_layer4.2.weight', 'TIN.residual_layer4.0.weight', 'lstm.weight_ih_l0_reverse', 'TIN.lstm.weight_ih_l0_reverse', 'TIN.residual_layer4.2.bias', 'lstm.bias_hh_l1_reverse', 'TIN.lstm.weight_hh_l0', 'attention_layer.w_value.bias', 'TIN.residual_layer1.0.bias', 'TIN.residual_layer4.0.bias', 'Syn_gcn.W.0.bias', 'TIN.lstm.weight_ih_l1_reverse', 'TIN.feature_fusion.3.weight', 'TIN.feature_fusion.3.bias', 'attention_layer.v.weight', 'size_embeddings.weight', 'lstm.weight_hh_l0_reverse', 'Sem_gcn.W.0.weight', 'lstm.bias_hh_l1', 'lstm.bias_hh_l0_reverse', 'fc.weight', 'attention_layer.linear_q.bias', 'TIN.residual_layer2.2.bias', 'TIN.GatedGCN.conv1.lin_r.bias', 'TIN.feature_fusion.0.weight', 'TIN.GatedGCN.conv3.lin_r.weight', 'Syn_gcn.W.1.bias', 'Sem_gcn.attn.linears.1.bias', 'TIN.lstm.bias_hh_l0_reverse', 'TIN.residual_layer2.0.bias', 'senti_classifier.weight', 'lstm.bias_ih_l0', 'TIN.GatedGCN.conv2.bias', 'TIN.residual_layer4.3.bias', 'TIN.GatedGCN.conv1.lin_l.weight', 'Sem_gcn.W.1.bias', 'attention_layer.w_query.bias', 'TIN.GatedGCN.conv2.lin_r.bias', 'TIN.lstm.bias_ih_l0', 'TIN.lstm.bias_hh_l1', 'TIN.residual_layer2.2.weight', 'TIN.residual_layer3.0.bias', 'TIN.lstm.bias_ih_l1', 'TIN.lstm.bias_hh_l1_reverse', 'lstm.weight_hh_l0', 'attention_layer.linear_q.weight', 'TIN.feature_fusion.0.bias', 'lstm.weight_ih_l0', 'TIN.residual_layer1.2.bias', 'TIN.residual_layer2.0.weight', 'lstm.weight_ih_l1_reverse', 'TIN.residual_layer2.3.bias', 'TIN.GatedGCN.conv3.lin_l.weight', 'attention_layer.w_query.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Train epoch 0: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 1: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.98it/s]
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

Train epoch 1: 100%|████████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 2: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.68it/s]
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

Train epoch 2: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 3: 100%|█████████████████████████| 27/27 [00:03<00:00,  8.45it/s]
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

Train epoch 3: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 4: 100%|█████████████████████████| 27/27 [00:05<00:00,  5.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        31.94        10.02        15.26          459
                   t        50.00         0.93         1.83          430

               micro        32.89         5.62         9.61          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        19.57         5.70         8.82          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        19.57         3.73         6.26          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         1.09         0.32         0.49          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro         1.09         0.21         0.35          483

Train epoch 4: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.78it/s]
Evaluate epoch 5: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o         0.00         0.00         0.00          459
                   t        36.92         5.58         9.70          430

               micro        36.92         2.70         5.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        31.58         1.90         3.58          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        31.58         1.24         2.39          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         0.00         0.00         0.00        316.0
                 NEG         0.00         0.00         0.00        142.0
                 NEU         0.00         0.00         0.00         25.0

               micro         0.00         0.00         0.00        483.0

Train epoch 5: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.77it/s]
Evaluate epoch 6: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.21it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        19.73        32.03        24.42          459
                   t        21.44        31.16        25.40          430

               micro        20.51        31.61        24.88          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS         8.21        12.66         9.96          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro         8.21         8.28         8.25          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS         1.23         1.90         1.49          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro         1.23         1.24         1.24          483

Train epoch 6: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.80it/s]
Evaluate epoch 7: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        28.74        15.90        20.48          459
                   t        31.22        30.93        31.07          430

               micro        30.29        23.17        26.26          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        38.64         5.38         9.44          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.64         3.52         6.45          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        20.45         2.85         5.00          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        20.45         1.86         3.42          483

Train epoch 7: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.75it/s]
Evaluate epoch 8: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.55it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        49.02         5.45         9.80          459
                   t        48.58        35.81        41.23          430

               micro        48.64        20.13        28.48          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        26.15         5.38         8.92          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        26.15         3.52         6.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        18.46         3.80         6.30          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        18.46         2.48         4.38          483

Train epoch 8: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.78it/s]
Evaluate epoch 9: 100%|█████████████████████████| 27/27 [00:03<00:00,  7.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        32.11         7.63        12.32          459
                   t        72.54        32.56        44.94          430

               micro        57.95        19.69        29.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        38.64         5.38         9.44          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        38.64         3.52         6.45          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        31.82         4.43         7.78          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        31.82         2.90         5.31          483

Train epoch 9: 100%|████████████████████████████| 49/49 [00:17<00:00,  2.78it/s]
Evaluate epoch 10: 100%|████████████████████████| 27/27 [00:03<00:00,  7.67it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        64.71        11.98        20.22          459
                   t        49.54        50.47        50.00          430

               micro        52.01        30.60        38.53          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        31.90        11.71        17.13          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        31.90         7.66        12.35          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.59        10.13        14.81          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.59         6.63        10.68          483

Train epoch 10: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.79it/s]
Evaluate epoch 11: 100%|████████████████████████| 27/27 [00:03<00:00,  7.61it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o       100.00         1.96         3.85          459
                   t        46.21        55.35        50.37          430

               micro        47.14        27.78        34.96          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        37.93         3.48         6.38          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        37.93         2.28         4.30          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.59         2.53         4.64          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.59         1.66         3.12          483

Train epoch 11: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.77it/s]
Evaluate epoch 12: 100%|████████████████████████| 27/27 [00:03<00:00,  7.27it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        37.79        45.53        41.30          459
                   t        48.87        40.23        44.13          430

               micro        42.12        42.97        42.54          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        15.16        26.58        19.31          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        15.16        17.39        16.20          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        15.16        26.58        19.31          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        15.16        17.39        16.20          483

Train epoch 12: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.76it/s]
Evaluate epoch 13: 100%|████████████████████████| 27/27 [00:03<00:00,  7.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        46.36        44.44        45.38          459
                   t        44.60        37.44        40.71          430

               micro        45.57        41.06        43.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        16.27        19.62        17.79          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        16.23        12.84        14.34          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        16.27        19.62        17.79          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        16.23        12.84        14.34          483

Train epoch 13: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.78it/s]
Evaluate epoch 14: 100%|████████████████████████| 27/27 [00:03<00:00,  7.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        53.97        29.63        38.26          459
                   t        46.76        23.49        31.27          430

               micro        50.64        26.66        34.93          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        24.06        10.13        14.25          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        23.88         6.63        10.37          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        24.06        10.13        14.25          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        23.88         6.63        10.37          483

Train epoch 14: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 15: 100%|████████████████████████| 27/27 [00:03<00:00,  7.83it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        30.97        53.38        39.20          459
                   t        55.31        23.02        32.51          430

               micro        35.46        38.70        37.01          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        14.23        12.34        13.22          316
                 NEG        10.53         1.41         2.48          142
                 NEU         0.00         0.00         0.00           25

               micro        13.99         8.49        10.57          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        14.23        12.34        13.22          316
                 NEG        10.53         1.41         2.48          142
                 NEU         0.00         0.00         0.00           25

               micro        13.99         8.49        10.57          483

Train epoch 15: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 16: 100%|████████████████████████| 27/27 [00:03<00:00,  8.09it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        61.89        32.90        42.96          459
                   t        51.61        22.33        31.17          430

               micro        57.44        27.78        37.45          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        42.39        12.34        19.12          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        42.39         8.07        13.57          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        42.39        12.34        19.12          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        42.39         8.07        13.57          483

Train epoch 16: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 17: 100%|████████████████████████| 27/27 [00:03<00:00,  7.80it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        38.24        49.24        43.05          459
                   t        58.90        22.33        32.38          430

               micro        42.71        36.22        39.20          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        29.41        18.99        23.08          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.99        12.42        17.39          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        27.45        17.72        21.54          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        27.05        11.59        16.23          483

Train epoch 17: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 18: 100%|████████████████████████| 27/27 [00:03<00:00,  8.41it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        73.97        23.53        35.70          459
                   t        65.67        10.23        17.71          430

               micro        71.36        17.10        27.59          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        50.91         8.86        15.09          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        50.91         5.80        10.41          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        50.91         8.86        15.09          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        50.91         5.80        10.41          483

Train epoch 18: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 19: 100%|████████████████████████| 27/27 [00:03<00:00,  8.11it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        43.36        43.36        43.36          459
                   t        66.12        18.60        29.04          430

               micro        48.10        31.38        37.99          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        31.05        18.67        23.32          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        30.89        12.22        17.51          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.42        17.09        21.34          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        28.27        11.18        16.02          483

Train epoch 19: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 20: 100%|████████████████████████| 27/27 [00:03<00:00,  8.23it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        68.94        24.18        35.81          459
                   t        59.85        36.74        45.53          430

               micro        63.29        30.26        40.94          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        42.18        19.62        26.78          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        42.18        12.84        19.68          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        40.82        18.99        25.92          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        40.82        12.42        19.05          483

Train epoch 20: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 21: 100%|████████████████████████| 27/27 [00:03<00:00,  6.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        40.80        59.91        48.54          459
                   t        42.12        53.49        47.13          430

               micro        41.39        56.81        47.89          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        29.11        41.46        34.20          316
                 NEG         6.06        12.68         8.20          142
                 NEU         0.00         0.00         0.00           25

               micro        19.95        30.85        24.23          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        28.44        40.51        33.42          316
                 NEG         6.06        12.68         8.20          142
                 NEU         0.00         0.00         0.00           25

               micro        19.54        30.23        23.74          483

Train epoch 21: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.80it/s]
Evaluate epoch 22: 100%|████████████████████████| 27/27 [00:03<00:00,  7.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.81        39.00        48.97          459
                   t        61.67        51.63        56.20          430

               micro        63.45        45.11        52.73          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        41.04        32.59        36.33          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        41.04        21.33        28.07          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        41.04        32.59        36.33          316
                 NEG         0.00         0.00         0.00          142
                 NEU         0.00         0.00         0.00           25

               micro        41.04        21.33        28.07          483

Train epoch 22: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 23: 100%|████████████████████████| 27/27 [00:03<00:00,  7.95it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        59.65        52.51        55.85          459
                   t        72.03        47.91        57.54          430

               micro        64.78        50.28        56.62          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        38.44        38.92        38.68          316
                 NEG        47.06         5.63        10.06          142
                 NEU         0.00         0.00         0.00           25

               micro        38.87        27.12        31.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        38.44        38.92        38.68          316
                 NEG        47.06         5.63        10.06          142
                 NEU         0.00         0.00         0.00           25

               micro        38.87        27.12        31.95          483

Train epoch 23: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.88it/s]
Evaluate epoch 24: 100%|████████████████████████| 27/27 [00:03<00:00,  7.98it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        63.20        46.41        53.52          459
                   t        61.70        53.95        57.57          430

               micro        62.41        50.06        55.56          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        34.16        43.67        38.33          316
                 NEG        66.67         1.41         2.76          142
                 NEU         0.00         0.00         0.00           25

               micro        34.40        28.99        31.46          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        33.91        43.35        38.06          316
                 NEG        66.67         1.41         2.76          142
                 NEU         0.00         0.00         0.00           25

               micro        34.15        28.78        31.24          483

Train epoch 24: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 25: 100%|████████████████████████| 27/27 [00:03<00:00,  7.26it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        45.74        59.69        51.80          459
                   t        42.79        66.98        52.22          430

               micro        44.18        63.22        52.01          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        30.32        42.41        35.36          316
                 NEG         7.33        28.87        11.70          142
                 NEU         0.00         0.00         0.00           25

               micro        17.48        36.23        23.58          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        30.09        42.09        35.09          316
                 NEG         6.80        26.76        10.84          142
                 NEU         0.00         0.00         0.00           25

               micro        17.08        35.40        23.05          483

Train epoch 25: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 26: 100%|████████████████████████| 27/27 [00:03<00:00,  7.73it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        41.20        70.37        51.97          459
                   t        60.00        55.12        57.45          430

               micro        47.50        62.99        54.16          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        35.18        44.30        39.22          316
                 NEG        13.09        17.61        15.02          142
                 NEU         0.00         0.00         0.00           25

               micro        28.01        34.16        30.78          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        34.92        43.99        38.94          316
                 NEG        13.09        17.61        15.02          142
                 NEU         0.00         0.00         0.00           25

               micro        27.84        33.95        30.60          483

Train epoch 26: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.92it/s]
Evaluate epoch 27: 100%|████████████████████████| 27/27 [00:03<00:00,  7.81it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        63.35        48.58        54.99          459
                   t        71.99        55.58        62.73          430

               micro        67.54        51.97        58.74          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        42.14        37.34        39.60          316
                 NEG        25.71         6.34        10.17          142
                 NEU         0.00         0.00         0.00           25

               micro        40.32        26.29        31.83          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        42.14        37.34        39.60          316
                 NEG        25.71         6.34        10.17          142
                 NEU         0.00         0.00         0.00           25

               micro        40.32        26.29        31.83          483

Train epoch 27: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 28: 100%|████████████████████████| 27/27 [00:03<00:00,  7.44it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.23        54.90        61.24          459
                   t        76.24        50.00        60.39          430

               micro        72.29        52.53        60.85          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        57.21        37.66        45.42          316
                 NEG        29.49        16.20        20.91          142
                 NEU         0.00         0.00         0.00           25

               micro        49.65        29.40        36.93          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        57.21        37.66        45.42          316
                 NEG        29.49        16.20        20.91          142
                 NEU         0.00         0.00         0.00           25

               micro        49.65        29.40        36.93          483

Train epoch 28: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 29: 100%|████████████████████████| 27/27 [00:03<00:00,  7.47it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        59.65        59.91        59.78          459
                   t        71.25        53.02        60.80          430

               micro        64.40        56.58        60.24          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        43.43        37.66        40.34          316
                 NEG        19.87        21.13        20.48          142
                 NEU         0.00         0.00         0.00           25

               micro        35.06        30.85        32.82          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        43.43        37.66        40.34          316
                 NEG        19.87        21.13        20.48          142
                 NEU         0.00         0.00         0.00           25

               micro        35.06        30.85        32.82          483

Train epoch 29: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 30: 100%|████████████████████████| 27/27 [00:03<00:00,  7.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        58.30        58.17        58.23          459
                   t        54.32        61.40        57.64          430

               micro        56.25        59.73        57.94          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        29.92        49.05        37.17          316
                 NEG        26.23        11.27        15.76          142
                 NEU         0.00         0.00         0.00           25

               micro        29.48        35.40        32.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        29.92        49.05        37.17          316
                 NEG        26.23        11.27        15.76          142
                 NEU         0.00         0.00         0.00           25

               micro        29.48        35.40        32.17          483

Train epoch 30: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 31: 100%|████████████████████████| 27/27 [00:03<00:00,  7.69it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.01        51.42        58.93          459
                   t        70.42        50.93        59.11          430

               micro        69.68        51.18        59.01          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        52.59        38.61        44.53          316
                 NEG        40.43        13.38        20.11          142
                 NEU        50.00         4.00         7.41           25

               micro        50.53        29.40        37.17          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        52.59        38.61        44.53          316
                 NEG        40.43        13.38        20.11          142
                 NEU        50.00         4.00         7.41           25

               micro        50.53        29.40        37.17          483

Train epoch 31: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.79it/s]
Evaluate epoch 32: 100%|████████████████████████| 27/27 [00:03<00:00,  7.68it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        74.40        47.49        57.98          459
                   t        69.67        53.95        60.81          430

               micro        71.88        50.62        59.41          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.81        37.97        44.53          316
                 NEG        47.92        16.20        24.21          142
                 NEU         0.00         0.00         0.00           25

               micro        52.57        29.61        37.88          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.81        37.97        44.53          316
                 NEG        47.92        16.20        24.21          142
                 NEU         0.00         0.00         0.00           25

               micro        52.57        29.61        37.88          483

Train epoch 32: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 33: 100%|████████████████████████| 27/27 [00:03<00:00,  7.77it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        71.43        51.20        59.64          459
                   t        80.59        44.42        57.27          430

               micro        75.27        47.92        58.56          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        52.34        35.44        42.26          316
                 NEG        48.84        14.79        22.70          142
                 NEU         0.00         0.00         0.00           25

               micro        51.75        27.54        35.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        52.34        35.44        42.26          316
                 NEG        48.84        14.79        22.70          142
                 NEU         0.00         0.00         0.00           25

               micro        51.75        27.54        35.95          483

Train epoch 33: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.90it/s]
Evaluate epoch 34: 100%|████████████████████████| 27/27 [00:03<00:00,  7.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.82        56.21        60.63          459
                   t        64.99        60.00        62.39          430

               micro        65.40        58.04        61.50          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        45.97        43.35        44.63          316
                 NEG        29.67        19.01        23.18          142
                 NEU        11.11         4.00         5.88           25

               micro        41.46        34.16        37.46          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        45.97        43.35        44.63          316
                 NEG        29.67        19.01        23.18          142
                 NEU        11.11         4.00         5.88           25

               micro        41.46        34.16        37.46          483

Train epoch 34: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 35: 100%|████████████████████████| 27/27 [00:03<00:00,  7.81it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.43        52.94        60.07          459
                   t        75.00        51.63        61.16          430

               micro        71.98        52.31        60.59          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        46.56        38.61        42.21          316
                 NEG        42.50        11.97        18.68          142
                 NEU        50.00        12.00        19.35           25

               micro        46.10        29.40        35.90          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        46.56        38.61        42.21          316
                 NEG        42.50        11.97        18.68          142
                 NEU        50.00        12.00        19.35           25

               micro        46.10        29.40        35.90          483

Train epoch 35: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.83it/s]
Evaluate epoch 36: 100%|████████████████████████| 27/27 [00:03<00:00,  7.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        60.18        59.26        59.71          459
                   t        66.04        49.30        56.46          430

               micro        62.61        54.44        58.24          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        43.17        37.97        40.40          316
                 NEG        43.90        12.68        19.67          142
                 NEU         0.00         0.00         0.00           25

               micro        42.99        28.57        34.33          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        43.17        37.97        40.40          316
                 NEG        43.90        12.68        19.67          142
                 NEU         0.00         0.00         0.00           25

               micro        42.99        28.57        34.33          483

Train epoch 36: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 37: 100%|████████████████████████| 27/27 [00:03<00:00,  7.74it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        69.34        54.68        61.14          459
                   t        71.34        55.58        62.48          430

               micro        70.30        55.12        61.79          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        47.19        45.25        46.20          316
                 NEG        64.00        11.27        19.16          142
                 NEU        28.57         8.00        12.50           25

               micro        48.06        33.33        39.36          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        47.19        45.25        46.20          316
                 NEG        64.00        11.27        19.16          142
                 NEU        28.57         8.00        12.50           25

               micro        48.06        33.33        39.36          483

Train epoch 37: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.89it/s]
Evaluate epoch 38: 100%|████████████████████████| 27/27 [00:03<00:00,  7.78it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.89        55.12        60.02          459
                   t        65.76        56.28        60.65          430

               micro        65.82        55.68        60.33          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        49.83        45.25        47.43          316
                 NEG        30.00         8.45        13.19          142
                 NEU         0.00         0.00         0.00           25

               micro        47.11        32.09        38.18          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        49.83        45.25        47.43          316
                 NEG        30.00         8.45        13.19          142
                 NEU         0.00         0.00         0.00           25

               micro        47.11        32.09        38.18          483

Train epoch 38: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 39: 100%|████████████████████████| 27/27 [00:03<00:00,  7.60it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.77        58.61        61.98          459
                   t        60.36        62.33        61.33          430

               micro        62.95        60.40        61.65          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        41.19        48.10        44.38          316
                 NEG        28.57        19.72        23.33          142
                 NEU         0.00         0.00         0.00           25

               micro        38.54        37.27        37.89          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        41.19        48.10        44.38          316
                 NEG        28.57        19.72        23.33          142
                 NEU         0.00         0.00         0.00           25

               micro        38.54        37.27        37.89          483

Train epoch 39: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.81it/s]
Evaluate epoch 40: 100%|████████████████████████| 27/27 [00:03<00:00,  7.76it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        60.66        60.13        60.39          459
                   t        69.25        56.05        61.95          430

               micro        64.38        58.16        61.11          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        56.90        41.77        48.18          316
                 NEG        27.43        21.83        24.31          142
                 NEU         0.00         0.00         0.00           25

               micro        47.25        33.75        39.37          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        56.47        41.46        47.81          316
                 NEG        27.43        21.83        24.31          142
                 NEU         0.00         0.00         0.00           25

               micro        46.96        33.54        39.13          483

Train epoch 40: 100%|███████████████████████████| 49/49 [00:16<00:00,  2.91it/s]
Evaluate epoch 41: 100%|████████████████████████| 27/27 [00:03<00:00,  7.84it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        70.97        52.72        60.50          459
                   t        62.28        57.21        59.64          430

               micro        66.30        54.89        60.06          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        51.49        43.67        47.26          316
                 NEG        58.54        16.90        26.23          142
                 NEU         0.00         0.00         0.00           25

               micro        52.26        33.54        40.86          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        51.49        43.67        47.26          316
                 NEG        58.54        16.90        26.23          142
                 NEU         0.00         0.00         0.00           25

               micro        52.26        33.54        40.86          483

Train epoch 41: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.83it/s]
Evaluate epoch 42: 100%|████████████████████████| 27/27 [00:03<00:00,  7.82it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        78.85        39.00        52.19          459
                   t        68.22        57.91        62.64          430

               micro        72.30        48.14        57.80          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        64.38        32.59        43.28          316
                 NEG        36.71        20.42        26.24          142
                 NEU        50.00         8.00        13.79           25

               micro        55.14        27.74        36.91          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        64.38        32.59        43.28          316
                 NEG        35.44        19.72        25.34          142
                 NEU        50.00         8.00        13.79           25

               micro        54.73        27.54        36.64          483

Train epoch 42: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.85it/s]
Evaluate epoch 43: 100%|████████████████████████| 27/27 [00:03<00:00,  7.72it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        71.51        52.51        60.55          459
                   t        66.58        58.84        62.47          430

               micro        68.90        55.57        61.52          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        49.44        42.09        45.47          316
                 NEG        29.47        19.72        23.63          142
                 NEU        66.67         8.00        14.29           25

               micro        44.41        33.75        38.35          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        49.44        42.09        45.47          316
                 NEG        29.47        19.72        23.63          142
                 NEU        66.67         8.00        14.29           25

               micro        44.41        33.75        38.35          483

Train epoch 43: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.82it/s]
Evaluate epoch 44: 100%|████████████████████████| 27/27 [00:03<00:00,  7.90it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        75.16        51.42        61.06          459
                   t        73.75        51.63        60.74          430

               micro        74.47        51.52        60.90          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.39        39.87        45.65          316
                 NEG        51.22        14.79        22.95          142
                 NEU         0.00         0.00         0.00           25

               micro        52.69        30.43        38.58          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        53.39        39.87        45.65          316
                 NEG        51.22        14.79        22.95          142
                 NEU         0.00         0.00         0.00           25

               micro        52.69        30.43        38.58          483

Train epoch 44: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.87it/s]
Evaluate epoch 45: 100%|████████████████████████| 27/27 [00:03<00:00,  7.64it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        68.97        56.64        62.20          459
                   t        71.63        59.30        64.89          430

               micro        70.26        57.93        63.50          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        48.20        46.52        47.34          316
                 NEG        36.47        21.83        27.31          142
                 NEU        33.33         4.00         7.14           25

               micro        45.55        37.06        40.87          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        48.20        46.52        47.34          316
                 NEG        36.47        21.83        27.31          142
                 NEU        33.33         4.00         7.14           25

               micro        45.55        37.06        40.87          483

Train epoch 45: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 46: 100%|████████████████████████| 27/27 [00:03<00:00,  7.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        76.14        50.76        60.92          459
                   t        75.47        56.51        64.63          430

               micro        75.80        53.54        62.76          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        59.13        43.04        49.82          316
                 NEG        36.76        17.61        23.81          142
                 NEU        33.33         4.00         7.14           25

               micro        53.82        33.54        41.33          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        59.13        43.04        49.82          316
                 NEG        36.76        17.61        23.81          142
                 NEU        33.33         4.00         7.14           25

               micro        53.82        33.54        41.33          483

Train epoch 46: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.86it/s]
Evaluate epoch 47: 100%|████████████████████████| 27/27 [00:03<00:00,  7.59it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        66.67        57.95        62.00          459
                   t        69.31        60.93        64.85          430

               micro        67.95        59.39        63.39          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        50.16        48.73        49.44          316
                 NEG        38.57        19.01        25.47          142
                 NEU        28.57         8.00        12.50           25

               micro        47.66        37.89        42.21          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        50.16        48.73        49.44          316
                 NEG        38.57        19.01        25.47          142
                 NEU        28.57         8.00        12.50           25

               micro        47.66        37.89        42.21          483

Train epoch 47: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.83it/s]
Evaluate epoch 48: 100%|████████████████████████| 27/27 [00:03<00:00,  7.63it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        72.29        52.29        60.68          459
                   t        69.34        58.37        63.38          430

               micro        70.75        55.23        62.03          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        50.35        45.25        47.67          316
                 NEG        40.43        13.38        20.11          142
                 NEU        50.00         4.00         7.41           25

               micro        48.95        33.75        39.95          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        50.35        45.25        47.67          316
                 NEG        40.43        13.38        20.11          142
                 NEU        50.00         4.00         7.41           25

               micro        48.95        33.75        39.95          483

Train epoch 48: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 49: 100%|████████████████████████| 27/27 [00:03<00:00,  7.56it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        65.83        57.08        61.14          459
                   t        68.21        58.37        62.91          430

               micro        66.97        57.71        61.99          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        46.96        43.99        45.42          316
                 NEG        29.90        20.42        24.27          142
                 NEU        25.00         4.00         6.90           25

               micro        42.57        34.99        38.41          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        46.62        43.67        45.10          316
                 NEG        29.90        20.42        24.27          142
                 NEU        25.00         4.00         6.90           25

               micro        42.32        34.78        38.18          483

Train epoch 49: 100%|███████████████████████████| 49/49 [00:17<00:00,  2.84it/s]
Evaluate epoch 50: 100%|████████████████████████| 27/27 [00:03<00:00,  7.65it/s]
Evaluation

---Aspect(Opinion) Term Extraction---

                type    precision       recall     f1-score      support
                   o        74.15        52.51        61.48          459
                   t        67.84        58.37        62.75          430

               micro        70.79        55.34        62.12          889


--- Aspect Sentiment Triplet Extraction ---

                type    precision       recall     f1-score      support
                 POS        53.03        44.30        48.28          316
                 NEG        34.33        16.20        22.01          142
                 NEU        25.00         4.00         6.90           25

               micro        48.96        33.95        40.10          483


---A sentiment is considered correct if the sentiment type and the two related entities are predicted correctly (in span and entity type)---

                type    precision       recall     f1-score      support
                 POS        52.65        43.99        47.93          316
                 NEG        34.33        16.20        22.01          142
                 NEU        25.00         4.00         6.90           25

               micro        48.66        33.75        39.85          483


Best F1 score: 42.21453287197232 at epoch 47