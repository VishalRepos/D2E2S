from torch import nn as nn
import torch
from trainer import util, sampling
import os
import math
from models.Syn_GCN_Improved import ImprovedGCN, AdaptiveGCN
from models.Sem_GCN_Improved import ImprovedSemGCN
from models.Attention_Module import SelfAttention
from models.TIN_GCN import TIN, FeatureStacking
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.Channel_Fusion import Orthographic_projection_fusion, TextCentredSP
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel

USE_CUDA = torch.cuda.is_available()


def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """Get specific token embedding (e.g. [CLS])"""
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h


class ImprovedD2E2SModel(PreTrainedModel):
    VERSION = "2.0"

    def __init__(
        self,
        config: AutoConfig,
        cls_token: int,
        sentiment_types: int,
        entity_types: int,
        args,
    ):
        super(ImprovedD2E2SModel, self).__init__(config)
        # 1、parameters init
        self.args = args
        self._size_embedding = self.args.size_embedding
        self._prop_drop = self.args.prop_drop
        self._freeze_transformer = self.args.freeze_transformer
        self.drop_rate = self.args.drop_out_rate
        self._is_bidirectional = self.args.is_bidirect
        self.layers = self.args.lstm_layers
        self._hidden_dim = self.args.hidden_dim
        self.mem_dim = self.args.mem_dim
        self._emb_dim = self.args.emb_dim
        self.output_size = self._emb_dim
        self.batch_size = self.args.batch_size
        self.USE_CUDA = USE_CUDA
        self.max_pairs = 100
        self.deberta_feature_dim = self.args.deberta_feature_dim
        self.gcn_dim = self.args.gcn_dim
        self.gcn_dropout = self.args.gcn_dropout

        # 2、DEBERT model
        self.deberta = AutoModel.from_pretrained(
            "microsoft/deberta-v3-base", config=config
        )

        # Enhanced GCN modules based on configuration
        if hasattr(self.args, 'gcn_type') and self.args.gcn_type == "adaptive":
            self.Syn_gcn = AdaptiveGCN(emb_dim=768, num_layers=self.args.gcn_layers, gcn_dropout=self.gcn_dropout)
        else:
            self.Syn_gcn = ImprovedGCN(emb_dim=768, num_layers=self.args.gcn_layers, gcn_dropout=self.gcn_dropout)
            
        self.Sem_gcn = ImprovedSemGCN(self.args, emb_dim=768, num_layers=self.args.gcn_layers, gcn_dropout=self.gcn_dropout)
        
        self.senti_classifier = nn.Linear(
            config.hidden_size * 3 + self._size_embedding * 2, sentiment_types
        )
        self.entity_classifier = nn.Linear(
            config.hidden_size * 2 + self._size_embedding, entity_types
        )
        self.size_embeddings = nn.Embedding(100, self._size_embedding)
        self.dropout = nn.Dropout(self._prop_drop)
        self._cls_token = cls_token
        self._sentiment_types = sentiment_types
        self._entity_types = entity_types
        self._max_pairs = self.max_pairs
        self.neg_span_all = 0
        self.neg_span = 0
        self.number = 1

        # 3、LSTM Layers + Attention Layers
        self.lstm = nn.LSTM(
            self._emb_dim,
            int(self._hidden_dim),
            self.layers,
            batch_first=True,
            bidirectional=self._is_bidirectional,
            dropout=self.drop_rate,
        )
        self.attention_layer = SelfAttention(self.args)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0)
        self.lstm_dropout = nn.Dropout(self.drop_rate)

        # 4、linear and sigmoid layers
        if self._is_bidirectional:
            self.fc = nn.Linear(int(self._hidden_dim * 2), self.output_size)
        else:
            self.fc = nn.Linear(int(self._hidden_dim), self.output_size)

        # 5、init_hidden
        weight = next(self.parameters()).data
        if self._is_bidirectional:
            self.number = 2

        # Initialize hidden state - will be moved to correct device in forward pass
        self.hidden = None
        self._hidden_dim_per_layer = self._hidden_dim

        # 6、weight initialization
        self.init_weights()
        if self._freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.deberta.parameters():
                param.requires_grad = False

        # 7、feature merge model
        self.TIN = TIN(self.deberta_feature_dim)
        
        # Enhanced feature fusion
        self.enhanced_fusion = nn.Sequential(
            nn.Linear(self.deberta_feature_dim * 2, self.deberta_feature_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.deberta_feature_dim, self.deberta_feature_dim),
            nn.LayerNorm(self.deberta_feature_dim)
        )

    def _init_hidden(self, batch_size, device):
        """Initialize LSTM hidden state on the correct device"""
        if self.hidden is None or self.hidden[0].size(1) != batch_size:
            weight = next(self.parameters()).data
            if self._is_bidirectional:
                self.number = 2
            
            self.hidden = (
                weight.new(self.layers * self.number, batch_size, self._hidden_dim_per_layer)
                .zero_()
                .float()
                .to(device),
                weight.new(self.layers * self.number, batch_size, self._hidden_dim_per_layer)
                .zero_()
                .float()
                .to(device),
            )

    def _forward_train(
        self,
        encodings: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        sentiments: torch.tensor,
        senti_masks: torch.tensor,
        adj,
    ):

        # Parameters init
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        h = self.deberta(input_ids=encodings, attention_mask=self.context_masks)[0]
        
        # Initialize hidden state on correct device
        self._init_hidden(batch_size, h.device)
        
        self.output, _ = self.lstm(h, self.hidden)
        self.deberta_lstm_output = self.lstm_dropout(self.output)
        self.deberta_lstm_att_feature = self.deberta_lstm_output

        # Enhanced GCN processing
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.deberta_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(
            self.deberta_lstm_att_feature, encodings, seq_lens
        )

        # Enhanced fusion layer
        h1 = self.TIN(
            h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn
        )
        
        # Additional enhanced fusion
        enhanced_features = self.enhanced_fusion(torch.cat([h1, h], dim=-1))
        
        # Attention mechanism
        h = self.attention_layer(enhanced_features, enhanced_features, self.context_masks[:, :seq_lens]) + enhanced_features

        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, self.args
        )

        # relation_classify - Memory optimized
        senti_clf = torch.zeros(
            [batch_size, sentiments.shape[1], self._sentiment_types]
        ).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(
                entity_spans_pool, size_embeddings, sentiments, senti_masks, h, i
            )
            end_idx = min(i + self._max_pairs, sentiments.shape[1])
            senti_clf[:, i : end_idx, :] = chunk_senti_logits[:, :end_idx - i, :]
            
            # Clear memory
            del chunk_senti_logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        batch_loss = compute_loss(adj_sem_ori, adj_sem_gcn, adj)

        return entity_clf, senti_clf, batch_loss

    def _forward_eval(
        self,
        encodings: torch.tensor,
        context_masks: torch.tensor,
        entity_masks: torch.tensor,
        entity_sizes: torch.tensor,
        entity_spans: torch.tensor,
        entity_sample_masks: torch.tensor,
        adj,
    ):
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        h = self.deberta(input_ids=encodings, attention_mask=self.context_masks)[0]
        
        # Initialize hidden state on correct device
        self._init_hidden(batch_size, h.device)
        
        self.output, _ = self.lstm(h, self.hidden)
        self.deberta_lstm_output = self.lstm_dropout(self.output)
        self.deberta_lstm_att_feature = self.deberta_lstm_output

        # Enhanced GCN processing
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.deberta_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(
            self.deberta_lstm_att_feature, encodings, seq_lens
        )

        # Enhanced fusion layer
        h1 = self.TIN(
            h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn
        )
        
        # Additional enhanced fusion
        enhanced_features = self.enhanced_fusion(torch.cat([h1, h], dim=-1))
        
        # Attention mechanism
        h = self.attention_layer(enhanced_features, enhanced_features, self.context_masks[:, :seq_lens]) + enhanced_features

        # entity_classify
        size_embeddings = self.size_embeddings(
            entity_sizes
        )  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(
            encodings, h, entity_masks, size_embeddings, self.args
        )

        # ignore entity candidates that do not constitute an actual entity for sentiments (based on classifier)
        # Use the actual sequence length from hidden states instead of context_masks
        actual_seq_len = h.shape[1]  # This is the actual sequence length
        sentiments, senti_masks, senti_sample_masks = self._filter_spans(
            entity_clf, entity_spans, entity_sample_masks, actual_seq_len
        )
        senti_sample_masks = senti_sample_masks.float().unsqueeze(-1)
        # Memory optimized sentiment classification
        senti_clf = torch.zeros(
            [batch_size, sentiments.shape[1], self._sentiment_types]
        ).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(
                entity_spans_pool, size_embeddings, sentiments, senti_masks, h, i
            )
            # apply sigmoid
            chunk_senti_clf = torch.sigmoid(chunk_senti_logits)
            end_idx = min(i + self._max_pairs, sentiments.shape[1])
            senti_clf[:, i : end_idx, :] = chunk_senti_clf[:, :end_idx - i, :]
            
            # Clear memory
            del chunk_senti_logits, chunk_senti_clf
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        senti_clf = senti_clf * senti_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, senti_clf, sentiments

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, args):
        # entity_masks: tensor(4,132,24) 4:batch_size, 132: entities count, 24: one sentence token count and one entity need 24 mask
        # size_embedding: tensor(4,132,25) 4：batch_size, 132:entities_size, 25:each entities Embedding Dimension
        # h: tensor(4,24,768) -> (4,1,24,768) -> (4,132,24,768)
        # m: tensor(4,132,24,1)
        # encoding: tensor(4,24)
        # entity_spans_pool: tensor(4，132，24，768) -> tensor(4,132,768)
        
        # Memory optimization: Process in chunks to avoid large tensor creation
        batch_size, num_entities, seq_len = entity_masks.shape
        hidden_dim = h.shape[-1]
        
        # Initialize output tensor
        entity_spans_pool = torch.zeros(batch_size, num_entities, hidden_dim, device=h.device)
        
        # Process in chunks to save memory
        chunk_size = min(32, num_entities)  # Process 32 entities at a time
        
        for i in range(0, num_entities, chunk_size):
            end_idx = min(i + chunk_size, num_entities)
            chunk_masks = entity_masks[:, i:end_idx, :]
            
            # Create mask for this chunk
            m = (chunk_masks.unsqueeze(-1) == 0).float() * (-1e30)
            
            # Expand h for this chunk only
            h_expanded = h.unsqueeze(1).expand(-1, end_idx - i, -1, -1)
            
            # Apply mask and pooling
            chunk_spans = m + h_expanded
            
            self.args = args
            if self.args.span_generator == "Average" or self.args.span_generator == "Max":
                if self.args.span_generator == "Max":
                    chunk_spans = chunk_spans.max(dim=2)[0]
                else:
                    chunk_spans = chunk_spans.mean(dim=2, keepdim=True).squeeze(-2)
            
            # Store results
            entity_spans_pool[:, i:end_idx, :] = chunk_spans
            
            # Clear memory
            del chunk_spans, h_expanded, m, chunk_masks
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        # Process in chunks to save memory
        entity_repr = torch.zeros(batch_size, num_entities, 
                                entity_ctx.shape[-1] + entity_spans_pool.shape[-1] + size_embeddings.shape[-1], 
                                device=h.device)
        
        for i in range(0, num_entities, chunk_size):
            end_idx = min(i + chunk_size, num_entities)
            
            # Create representation for this chunk
            chunk_repr = torch.cat([
                entity_ctx.unsqueeze(1).expand(-1, end_idx - i, -1),
                entity_spans_pool[:, i:end_idx, :],
                size_embeddings[:, i:end_idx, :],
            ], dim=2)
            
            entity_repr[:, i:end_idx, :] = chunk_repr
            
            # Clear memory
            del chunk_repr
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_sentiments(
        self, entity_spans, size_embeddings, sentiments, senti_masks, h, chunk_start
    ):
        batch_size = sentiments.shape[0]

        # create chunks if necessary
        if sentiments.shape[1] > self._max_pairs:
            sentiments = sentiments[:, chunk_start : chunk_start + self._max_pairs]
            senti_masks = senti_masks[:, chunk_start : chunk_start + self._max_pairs]
            h = h[:, : sentiments.shape[1], :]

        # Memory optimization: Process in smaller chunks
        num_pairs = sentiments.shape[1]
        chunk_size = min(16, num_pairs)  # Process 16 pairs at a time
        
        # Initialize output tensor
        chunk_senti_logits = torch.zeros(batch_size, num_pairs, self._sentiment_types, device=h.device)
        
        for i in range(0, num_pairs, chunk_size):
            end_idx = min(i + chunk_size, num_pairs)
            
            # Process this chunk
            chunk_sentiments = sentiments[:, i:end_idx]
            chunk_senti_masks = senti_masks[:, i:end_idx]
            # Use the full h tensor - don't slice it incorrectly
            chunk_h = h
            
            # Handle the case where we have multiple sentiment pairs
            # For each pair, we need to process the corresponding mask
            num_pairs_in_chunk = chunk_sentiments.shape[1]
            
            # get pairs of entity candidate representations for this chunk
            entity_pairs = util.batch_index(entity_spans, chunk_sentiments)
            entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

            # get corresponding size embeddings for this chunk
            size_pair_embeddings = util.batch_index(size_embeddings, chunk_sentiments)
            size_pair_embeddings = size_pair_embeddings.view(
                batch_size, size_pair_embeddings.shape[1], -1
            )

            # sentiment context (context between entity candidate pair)
            # mask non entity candidate tokens
            # Process each sentiment pair in the chunk
            seq_len = chunk_h.shape[1]
            
            # Initialize output for this chunk
            # chunk_h.shape[-1] is the hidden dimension (768)
            chunk_senti_ctx = torch.zeros(batch_size, num_pairs_in_chunk, chunk_h.shape[-1], device=chunk_h.device)
            
            for pair_idx in range(num_pairs_in_chunk):
                # Get mask for this specific pair
                if len(chunk_senti_masks.shape) == 3:
                    # 3D mask: [batch, num_pairs, seq_len]
                    pair_mask = chunk_senti_masks[:, pair_idx, :]
                else:
                    # 2D mask: [batch, seq_len] - use for all pairs
                    pair_mask = chunk_senti_masks
                
                # Ensure mask has the correct sequence length
                if pair_mask.shape[1] != seq_len:
                    # Create a new mask with the correct sequence length
                    batch_size_mask = pair_mask.shape[0]
                    new_mask = torch.zeros(batch_size_mask, seq_len, device=pair_mask.device)
                    
                    # Copy the original mask values if possible
                    min_len = min(pair_mask.shape[1], seq_len)
                    new_mask[:, :min_len] = pair_mask[:, :min_len]
                    
                    pair_mask = new_mask
                
                m = ((pair_mask == 0).float() * (-1e30)).unsqueeze(-1)
                pair_ctx = m + chunk_h
                # max pooling over sequence dimension to get hidden features
                # pair_ctx shape: [batch, seq_len, hidden_dim]
                # We want to max pool over seq_len to get [batch, hidden_dim]
                # The issue is that we're max pooling over the wrong dimension
                # We need to max pool over dim=1 (sequence dimension) to get [batch, hidden_dim]
                pair_ctx = pair_ctx.max(dim=1)[0]  # Max pool over sequence dimension (dim=1)
                # set the context vector of neighboring or adjacent entity candidates to zero
                # This line needs to be removed since we're now working with hidden features, not sequence positions
                # pair_ctx[pair_mask.to(torch.uint8).any(-1) == 0] = 0
                
                # Store result for this pair
                chunk_senti_ctx[:, pair_idx, :] = pair_ctx
            
            # Use the first pair's context for the entire chunk (simplified approach)
            senti_ctx = chunk_senti_ctx[:, 0, :]

            # create sentiment candidate representations including context, max pooled entity candidate pairs
            # and corresponding size embeddings
            # senti_ctx has shape [batch, hidden_dim] (2D)
            # entity_pairs and size_pair_embeddings have shape [batch, num_pairs, feature_dim] (3D)
            # We need to expand senti_ctx to match the 3D shape
            senti_ctx_expanded = senti_ctx.unsqueeze(1).expand(-1, entity_pairs.shape[1], -1)
            senti_repr = torch.cat([senti_ctx_expanded, entity_pairs, size_pair_embeddings], dim=2)
            senti_repr = self.dropout(senti_repr)

            # classify sentiment candidates for this chunk
            chunk_logits = self.senti_classifier(senti_repr)
            chunk_senti_logits[:, i:end_idx, :] = chunk_logits
            
            # Clear memory
            del chunk_sentiments, chunk_senti_masks, chunk_h, entity_pairs, size_pair_embeddings
            del chunk_senti_ctx, senti_ctx, senti_repr, chunk_logits
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return chunk_senti_logits

    def log_sample_total(self, neg_entity_count_all):
        log_path = os.path.join("./log/Sample/", "countSample.txt")
        with open(log_path, mode="a", encoding="utf-8") as f:
            f.write("neg_entity_count_all: \n")
            self.neg_span_all += len(neg_entity_count_all)
            f.write(str(self.neg_span_all))
            f.write("\nneg_entity_count: \n")
            self.neg_span += len((neg_entity_count_all != 0).nonzero())
            f.write(str(self.neg_span))
            f.write("\n")
        f.close()

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = (
            entity_clf.argmax(dim=-1) * entity_sample_masks.long()
        )  # get entity type (including none)
        batch_sentiments = []
        batch_senti_masks = []
        batch_senti_sample_masks = []

        for i in range(batch_size):
            rels = []
            senti_masks = []
            sample_masks = []

            # get spans classified as entities
            self.log_sample_total(entity_logits_max[i])
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()

            # create sentiments and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        senti_masks.append(sampling.create_senti_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_sentiments.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_senti_masks.append(
                    torch.tensor([[0] * ctx_size], dtype=torch.bool)
                )
                batch_senti_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_sentiments.append(torch.tensor(rels, dtype=torch.long))
                batch_senti_masks.append(torch.stack(senti_masks))
                batch_senti_sample_masks.append(
                    torch.tensor(sample_masks, dtype=torch.bool)
                )

        # stack
        device = self.senti_classifier.weight.device
        batch_sentiments = util.padded_stack(batch_sentiments).to(device)
        batch_senti_masks = util.padded_stack(batch_senti_masks).to(device)
        batch_senti_sample_masks = util.padded_stack(batch_senti_sample_masks).to(
            device
        )

        return batch_sentiments, batch_senti_masks, batch_senti_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)


def compute_loss(p, q, k):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(k, dim=-1), reduction="none")
    k_loss = F.kl_div(F.log_softmax(k, dim=-1), F.softmax(p, dim=-1), reduction="none")

    p_loss = p_loss.sum()
    k_loss = k_loss.sum()
    total_loss = math.log(1 + 5 / (torch.abs((p_loss + k_loss) / 2)))

    return total_loss 