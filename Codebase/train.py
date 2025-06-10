import argparse
import math
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch
import transformers
from torch.optim import optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW
from transformers import AutoTokenizer, AutoConfig

from Parameter import train_argparser
from models.D2E2S_Model import D2E2SModel
from models.General import set_seed
from trainer import util, sampling
from trainer.baseTrainer import BaseTrainer
from trainer.entities import Dataset
from trainer.evaluator import Evaluator
from trainer.input_reader import JsonInputReader
from trainer.loss import D2E2SLoss
import warnings

warnings.filterwarnings("ignore")


class D2E2S_Trainer(BaseTrainer):
    def __init__(self, args: argparse.Namespace):
        # Existing initialization
        super().__init__(args)
        self._tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-base")
        self._predictions_path = os.path.join(
            self._log_path_predict, "predicted_%s_epoch_%s.json"
        )
        self._examples_path = os.path.join(
            self._log_path_predict, "sample_%s_%s_epoch_%s.html"
        )
        os.makedirs(self._log_path_result)
        os.makedirs(self._log_path_predict)
        
        # Add visualization paths
        self._attention_viz_path = os.path.join(self._log_path, "attention_viz")
        os.makedirs(self._attention_viz_path, exist_ok=True)
        
        self.max_pair_f1 = 40
        self.result_path = os.path.join(
            self._log_path_result, "result{}.txt".format(self.args.max_span_size)
        )
        
    def _preprocess(self, args, input_reader_cls, types_path, train_path, test_path):

        train_label, test_label = "train", "test"
        # create log csv files
        self._init_train_logging(train_label)
        self._init_eval_logging(test_label)

        # loading data
        input_reader = input_reader_cls(
            types_path,
            self._tokenizer,
            args.neg_entity_count,
            args.neg_triple_count,
            args.max_span_size,
        )
        input_reader.read({train_label: train_path, test_label: test_path})
        train_dataset = input_reader.get_dataset(train_label)

        # preprocess
        train_sample_count = train_dataset.sentence_count
        updates_epoch = train_sample_count // args.batch_size
        updates_total = updates_epoch * args.epochs

        print("   ", self.args.dataset, "  ", self.args.max_span_size)
        return input_reader, updates_total, updates_epoch

    def _train(
        self, train_path: str, test_path: str, types_path: str, input_reader_cls
    ):
        args = self.args

        # set seed
        set_seed(args.seed)

        # Create attention visualization directory
        self._attention_viz_path = os.path.join(self._log_path, "attention_viz")
        os.makedirs(self._attention_viz_path, exist_ok=True)

        train_label, test_label = "train", "test"
        input_reader, updates_total, updates_epoch = self._preprocess(
            args, input_reader_cls, types_path, train_path, test_path
        )
        train_dataset = input_reader.get_dataset(train_label)
        test_dataset = input_reader.get_dataset(test_label)

        # load model
        config = AutoConfig.from_pretrained("microsoft/deberta-v3-base")
        # Enable attention outputs
        config.output_attentions = True
        config.use_cache = False  # Needed for attention outputs

        model = D2E2SModel.from_pretrained(
            self.args.pretrained_deberta_name,
            config=config,
            cls_token=self._tokenizer.convert_tokens_to_ids("[CLS]"),
            sentiment_types=input_reader.sentiment_type_count - 1,
            entity_types=input_reader.entity_type_count,
            args=args,
        )
        model.to(args.device)
        # create optimizer
        optimizer_params = self._get_optimizer_params(model)
        optimizer = AdamW(
            optimizer_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
            correct_bias=False,
        )
        # create scheduler
        scheduler = transformers.get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.lr_warmup * updates_total,
            num_training_steps=updates_total,
        )

        # create loss function
        entity_criterion = torch.nn.CrossEntropyLoss(reduction="none")
        senti_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
        compute_loss = D2E2SLoss(
            senti_criterion,
            entity_criterion,
            model,
            optimizer,
            scheduler,
            args.max_grad_norm,
        )
        # eval validation set
        if args.init_eval:
            self._eval(model, test_dataset, input_reader, 0, updates_epoch)

        # train
        for epoch in range(args.epochs):
            # train epoch
            self.train_epoch(
                model, compute_loss, optimizer, train_dataset, updates_epoch, epoch
            )

            # eval validation sets
            if not args.final_eval or (epoch == args.epochs - 1):
                # print(epoch)
                self._eval(model, test_dataset, input_reader, epoch + 1, updates_epoch)

    def train_epoch(
        self,
        model: torch.nn.Module,
        compute_loss: D2E2SLoss,
        optimizer: optimizer,
        dataset: Dataset,
        updates_epoch: int,
        epoch: int,
    ):
        # Create data loader
        dataset.switch_mode(Dataset.TRAIN_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        model.zero_grad()

        iteration = 0
        total = dataset.sentence_count // self.args.batch_size
        for batch_idx, batch in enumerate(tqdm(data_loader, total=total, desc=f"Train epoch {epoch}")):
            model.train()
            batch = util.to_device(batch, self.args.device)

            # Enable attention visualization periodically (every 50 batches)
            visualize_attention = batch_idx % 50 == 0
            if visualize_attention:
                model.store_attention_weights(True)

            # Forward step
            entity_logits, senti_logits, batch_loss = model(
                encodings=batch["encodings"],
                context_masks=batch["context_masks"],
                entity_masks=batch["entity_masks"],
                entity_sizes=batch["entity_sizes"],
                sentiments=batch["rels"],
                senti_masks=batch["senti_masks"],
                adj=batch["adj"],
            )

            # Visualize attention if enabled
            if visualize_attention:
                print(f"\nGenerating attention visualization for epoch {epoch} batch {batch_idx}")
                tokens = self._tokenizer.convert_ids_to_tokens(
                    batch["encodings"][0].cpu().numpy()
                )
                if hasattr(model, "attention_weights"):
                    # Log attention weights for debugging
                    for k, v in model.attention_weights.items():
                        if v is not None:
                            print(f"{k} attention shape: {v.shape}")
                            
                    # Generate visualizations
                    viz_path = os.path.join(
                        self._attention_viz_path,
                        f"train_epoch{epoch}_batch{batch_idx}"
                    )
                    print(f"Saving attention visualizations to: {viz_path}")
                    
                    model.visualizer.visualize_model_attention(
                        model.attention_weights.get('deberta'),
                        model.attention_weights.get('gcn_sem'),
                        model.attention_weights.get('gcn_syn'),
                        tokens,
                        save_prefix=viz_path
                    )
                else:
                    print("No attention weights found in model")

            # Compute loss and optimize parameters
            epoch_loss = compute_loss.compute(
                entity_logits=entity_logits,
                senti_logits=senti_logits,
                batch_loss=batch_loss,
                senti_types=batch["senti_types"],
                entity_types=batch["entity_types"],
                entity_sample_masks=batch["entity_sample_masks"],
                senti_sample_masks=batch["senti_sample_masks"],
            )

            # Logging
            iteration += 1
            global_iteration = epoch * updates_epoch + iteration

            if global_iteration % self.args.train_log_iter == 0:
                self._log_train(
                    optimizer,
                    epoch_loss,
                    epoch,
                    iteration,
                    global_iteration,
                    dataset.label,
                )

        return iteration

    def _log_train(
        self,
        optimizer: optimizer,
        loss: float,
        epoch: int,
        iteration: int,
        global_iteration: int,
        label: str,
    ):
        # average loss
        avg_loss = loss / self.args.batch_size
        # get current learning rate
        lr = self._get_lr(optimizer)[0]

        # to_csv
        # log to csv
        self._log_csv(label, "loss", loss, epoch, iteration, global_iteration)
        self._log_csv(label, "loss_avg", avg_loss, epoch, iteration, global_iteration)
        self._log_csv(label, "lr", lr, epoch, iteration, global_iteration)

        # log to tensorboard
        self._log_tensorboard(label, "loss", loss, global_iteration)
        self._log_tensorboard(label, "loss_avg", avg_loss, global_iteration)
        self._log_tensorboard(label, "lr", lr, global_iteration)

    def _eval(
        self,
        model: torch.nn.Module,
        dataset: Dataset,
        input_reader: JsonInputReader,
        epoch: int = 0,
        updates_epoch: int = 0,
        iteration: int = 0,
    ):
        # Create evaluator
        evaluator = Evaluator(
            dataset,
            input_reader,
            self._tokenizer,
            self.args.sen_filter_threshold,
            self._predictions_path,
            self._examples_path,
            self.args.example_count,
            epoch,
            dataset.label,
        )

        # Create data loader
        dataset.switch_mode(Dataset.EVAL_MODE)
        data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.args.sampling_processes,
            collate_fn=sampling.collate_fn_padding,
        )

        with torch.no_grad():
            model.eval()
            
            # Step 1: Enable attention storage before evaluation
            model.enable_attention_storage()
            print(f"Enabled attention storage for evaluation epoch {epoch}")

            # Iterate batches
            total = math.ceil(dataset.sentence_count / self.args.batch_size)
            for batch_idx, batch in enumerate(tqdm(data_loader, total=total, desc=f"Evaluate epoch {epoch}")):
                # Move batch to selected device
                batch = util.to_device(batch, self.args.device)

                # Step 2: Run forward pass (eval mode)
                result = model(
                    encodings=batch["encodings"],
                    context_masks=batch["context_masks"],
                    entity_masks=batch["entity_masks"],
                    entity_sizes=batch["entity_sizes"],
                    entity_spans=batch["entity_spans"],
                    entity_sample_masks=batch["entity_sample_masks"],
                    evaluate=True,
                    adj=batch["adj"],
                )
                entity_clf, senti_clf, rels = result

                # Step 3: Access attention weights and create visualizations periodically
                if batch_idx % 20 == 0:  # Visualize every 20 batches
                    attention_weights = model.get_attention_weights()
                    if attention_weights:
                        print(f"Creating visualizations for batch {batch_idx}")
                        # Get tokens for visualization
                        tokens = self._tokenizer.convert_ids_to_tokens(
                            batch["encodings"][0].cpu().numpy()
                        )
                        
                        # Step 4: Create and save visualizations
                        model.visualizer.visualize_model_attention(
                            attention_weights.get('deberta'),
                            attention_weights.get('gcn_sem'),
                            attention_weights.get('gcn_syn'),
                            tokens,
                            save_prefix=f"epoch{epoch}_batch{batch_idx}"
                        )
                        model.clear_attention_weights()  # Clear for next batch

                # Evaluate batch
                evaluator.eval_batch(entity_clf, senti_clf, rels, batch)

            # Disable attention storage after evaluation
            model.disable_attention_storage()

            global_iteration = epoch * updates_epoch + iteration
            ner_eval, senti_eval, senti_nec_eval = evaluator.compute_scores()
            self._log_filter_file(ner_eval, senti_eval, evaluator, epoch)

        self._log_eval(
            *ner_eval,
            *senti_eval,
            *senti_nec_eval,
            epoch,
            iteration,
            global_iteration,
            dataset.label
        )

    def _log_filter_file(self, ner_eval, senti_eval, evaluator, epoch):
        f1 = float(senti_eval[2])
        if self.max_pair_f1 < f1:
            columns = [
                "mic_precision",
                "mic_recall",
                "mic_f1_score",
                "mac_precision",
                "mac_recall",
                "mac_f1_score",
            ]
            ner_dic = {
                "mic_precision": 0.0,
                "mic_recall": 0.0,
                "mic_f1_score": 0.0,
                "mac_precision": 0.0,
                "mac_recall": 0.0,
                "mac_f1_score": 0.0,
            }
            senti_dic = {
                "mic_precision": 0.0,
                "mic_recall": 0.0,
                "mic_f1_score": 0.0,
                "mac_precision": 0.0,
                "mac_recall": 0.0,
                "mac_f1_score": 0.0,
            }
            # for inx, val in enumerate(ner_eval):
            #     ner_dic[columns[inx]] = val
            for inx, val in enumerate(senti_eval):
                senti_dic[columns[inx]] = val
            self.max_pair_f1 = f1
            with open(self.result_path, mode="a", encoding="utf-8") as f:
                w_str = "No. {} ：....\n".format(epoch)
                f.write(w_str)
                f.write("ner_entity: \n")
                f.write(str(ner_dic))
                f.write("\n rec: \n")
                f.write(str(senti_dic))
                f.write("\n")
            try:
                fileNames = os.listdir(self._log_path_predict)
                # print(fileNames)
                for i in fileNames:
                    os.remove(os.path.join(self._log_path_predict, i))
            except BaseException:
                print(BaseException)
            if self.args.store_predictions:
                evaluator.store_predictions()

            if self.args.store_examples:
                evaluator.store_examples()

    def _get_optimizer_params(self, model):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_params = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        return optimizer_params


if __name__ == "__main__":
    arg_parser = train_argparser()
    trainer = D2E2S_Trainer(arg_parser)
    trainer._train(
        train_path=arg_parser.dataset_file["train"],
        test_path=arg_parser.dataset_file["test"],
        types_path=arg_parser.dataset_file["types_path"],
        input_reader_cls=JsonInputReader,
    )
