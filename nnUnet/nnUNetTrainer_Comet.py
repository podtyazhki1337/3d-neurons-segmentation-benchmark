"""
nnUNetTrainer_Comet.py — Custom nnU-Net trainer with Comet ML logging.

Drop this file into:
    nnUNet/nnunetv2/training/nnUNetTrainer/nnUNetTrainer_Comet.py

All built-in nnU-Net augmentations remain active (rotation, scaling,
elastic deform, gamma, mirror, noise, blur) — this is the standard
pipeline from the Nature Methods paper.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainer_Comet(nnUNetTrainer):
    """nnUNetTrainer with Comet ML experiment tracking."""

    def __init__(self, plans: dict, configuration: str, fold: int,
                 dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.experiment = None
        self._epoch_start_time = None

    # ── Comet setup ──────────────────────────────────────────────────────

    def _init_comet(self):
        """Initialize Comet experiment. Called once at training start."""
        try:
            from comet_ml import Experiment

            api_key   = os.environ.get("COMET_API_KEY", "")
            project   = os.environ.get("NNUNET_COMET_PROJECT", "nnunet-benchmark")
            workspace = os.environ.get("NNUNET_COMET_WORKSPACE", "")

            if not api_key:
                self.print_to_log_file("⚠️  COMET_API_KEY not set, Comet logging disabled.")
                return

            kwargs = {"api_key": api_key, "project_name": project}
            if workspace:
                kwargs["workspace"] = workspace

            self.experiment = Experiment(**kwargs)

            # Log nnU-Net plans as parameters
            self.experiment.log_parameters({
                "model": "nnU-Net",
                "configuration": self.configuration_name,
                "fold": self.fold,
                "num_epochs": self.num_epochs,
                "batch_size": self.configuration_manager.batch_size,
                "patch_size": str(self.configuration_manager.patch_size),
                "architecture": str(self.plans_manager.plans_name),
                "trainer_class": self.__class__.__name__,
                "augmentation": "DEFAULT (nnU-Net built-in)",
            })

            # Log plans JSON as text artifact
            try:
                plans_dict = self.plans_manager.plans
                plans_str = json.dumps(plans_dict, indent=2, default=str)
                self.experiment.log_text(plans_str, metadata={"name": "nnUNetPlans.json"})
            except Exception:
                pass

            # Log dataset.json
            ds_str = json.dumps(self.dataset_json, indent=2)
            self.experiment.log_text(ds_str, metadata={"name": "dataset.json"})

            # Log extra metadata passed via env (set by train.py)
            extra_keys = [
                "NNUNET_DATASET_ID", "NNUNET_MODALITY", "NNUNET_ANIMAL",
                "NNUNET_VOXEL_SIZE", "NNUNET_AUG_PRESET", "NNUNET_SEG_APPROACH",
                "NNUNET_SEED",
            ]
            extra = {k: os.environ.get(k, "") for k in extra_keys if os.environ.get(k)}
            if extra:
                self.experiment.log_parameters(extra)

            self.print_to_log_file(f"✅  Comet experiment: {self.experiment.url}")

        except ImportError:
            self.print_to_log_file("⚠️  comet_ml not installed, Comet logging disabled.")
        except Exception as e:
            self.print_to_log_file(f"⚠️  Comet init failed: {e}")

    # ── Override training hooks ──────────────────────────────────────────

    def on_train_start(self):
        super().on_train_start()
        self._init_comet()

    def on_epoch_start(self):
        super().on_epoch_start()
        self._epoch_start_time = time.time()

    def on_epoch_end(self):
        super().on_epoch_end()

        if self.experiment is None:
            return

        epoch = self.current_epoch
        epoch_time = time.time() - self._epoch_start_time if self._epoch_start_time else 0

        try:
            # Log losses (MetaLogger API: get_value(key, step=-1))
            self.experiment.log_metric("train_loss",
                self.logger.get_value("train_losses", step=-1), epoch=epoch)
            self.experiment.log_metric("val_loss",
                self.logger.get_value("val_losses", step=-1), epoch=epoch)

            # Log learning rate
            self.experiment.log_metric("lr",
                self.logger.get_value("lrs", step=-1), epoch=epoch)

            # Log epoch time
            self.experiment.log_metric("epoch_time_sec", epoch_time, epoch=epoch)

            # Log EMA foreground dice
            try:
                ema_dc = self.logger.get_value("ema_fg_dice", step=-1)
                if isinstance(ema_dc, (list, np.ndarray)):
                    for i, v in enumerate(ema_dc):
                        self.experiment.log_metric(f"ema_fg_dice_class_{i}", float(v), epoch=epoch)
                    self.experiment.log_metric("ema_fg_dice_mean",
                        float(np.mean(ema_dc)), epoch=epoch)
                else:
                    self.experiment.log_metric("ema_fg_dice", float(ema_dc), epoch=epoch)
            except Exception:
                pass

            # Log per-class dice
            try:
                mean_dc = self.logger.get_value("dice_per_class_or_region", step=-1)
                if isinstance(mean_dc, (list, np.ndarray)):
                    for i, v in enumerate(mean_dc):
                        self.experiment.log_metric(f"val_dice_class_{i}", float(v), epoch=epoch)
                    self.experiment.log_metric("val_dice_mean",
                        float(np.mean(mean_dc)), epoch=epoch)
            except Exception:
                pass

            # Log mean foreground dice
            try:
                mfg = self.logger.get_value("mean_fg_dice", step=-1)
                self.experiment.log_metric("mean_fg_dice", float(mfg), epoch=epoch)
            except Exception:
                pass

        except Exception as e:
            self.print_to_log_file(f"⚠️  Comet logging error: {e}")

    def on_train_end(self):
        super().on_train_end()
        if self.experiment is not None:
            self.experiment.log_parameter("model_output_dir", str(self.output_folder))
            self.experiment.end()
            self.print_to_log_file("✅  Comet experiment ended.")