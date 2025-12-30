import torch
import torch.nn.functional as F

class FocalLoss(torch.nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='none'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss

class D2E2SLoss():
    def __init__(self, senti_criterion, entity_criterion, model, optimizer, scheduler, max_grad_norm, accumulation_steps=4):
        self._senti_criterion = senti_criterion
        # Use standard CrossEntropy instead of Focal Loss to prevent NaN
        self._entity_criterion = entity_criterion  # Use original criterion
        self._original_entity_criterion = entity_criterion
        self._model = model
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._max_grad_norm = max_grad_norm
        self._accumulation_steps = accumulation_steps
        self._current_step = 0
        self._label_smoothing = 0.0  # Disable label smoothing to prevent NaN

    def compute(self, entity_logits, senti_logits, batch_loss, entity_types, senti_types, entity_sample_masks, senti_sample_masks):
        # Check inputs for NaN
        if torch.isnan(entity_logits).any():
            print("⚠️ NaN in entity_logits!")
            self._model.zero_grad()
            return torch.tensor(0.0).to(entity_logits.device)
        if torch.isnan(senti_logits).any():
            print("⚠️ NaN in senti_logits!")
            self._model.zero_grad()
            return torch.tensor(0.0).to(senti_logits.device)
        
        # term loss
        entity_logits = entity_logits.view(-1, entity_logits.shape[-1])
        entity_types = entity_types.view(-1)
        entity_sample_masks = entity_sample_masks.view(-1).float()

        entity_loss = self._entity_criterion(entity_logits, entity_types)
        entity_loss = (entity_loss * entity_sample_masks).sum() / (entity_sample_masks.sum() + 1e-8)

        # sentiment loss (no label smoothing)
        senti_sample_masks = senti_sample_masks.view(-1).float()
        senti_count = senti_sample_masks.sum()

        if senti_count.item() != 0:
            senti_logits = senti_logits.view(-1, senti_logits.shape[-1])
            senti_types = senti_types.view(-1, senti_types.shape[-1])

            senti_loss = self._senti_criterion(senti_logits, senti_types)
            senti_loss = senti_loss.sum(-1) / senti_loss.shape[-1]
            senti_loss = (senti_loss * senti_sample_masks).sum() / (senti_count + 1e-8)

            # No batch_loss multiplier
            train_loss = entity_loss + senti_loss
        else:
            train_loss = entity_loss
            senti_loss = torch.tensor(0.0).to(entity_loss.device)

        # Check for NaN/Inf
        if torch.isnan(train_loss) or torch.isinf(train_loss):
            print(f"⚠️ WARNING: Loss is NaN/Inf! Skipping this batch.")
            print(f"  entity_loss: {entity_loss.item() if not torch.isnan(entity_loss) else 'nan'}")
            print(f"  senti_loss: {senti_loss.item() if not torch.isnan(senti_loss) else 'nan'}")
            print(f"  batch_loss: {batch_loss if isinstance(batch_loss, float) else batch_loss.item()}")
            self._model.zero_grad()
            return torch.tensor(0.0).to(entity_loss.device)

        # Clip loss
        train_loss = torch.clamp(train_loss, max=10.0)
        
        # Scale loss for gradient accumulation
        train_loss = train_loss / self._accumulation_steps
        train_loss.backward()
        
        self._current_step += 1
        
        # Only update weights every accumulation_steps
        if self._current_step % self._accumulation_steps == 0:
            # Check for NaN gradients before clipping
            has_nan_grad = False
            for name, param in self._model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                        print(f"⚠️ WARNING: NaN/Inf gradient in {name}")
                        has_nan_grad = True
                        break
            
            if has_nan_grad:
                print("⚠️ Skipping optimizer step due to NaN gradients")
                self._model.zero_grad()
                return 0.0
            
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._max_grad_norm)
            self._optimizer.step()
            self._scheduler.step()
            self._model.zero_grad()
            torch.cuda.empty_cache()  # Clear cache after optimizer step
        
        return train_loss.item() * self._accumulation_steps  # Return unscaled loss for logging