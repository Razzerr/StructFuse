import hydra
from omegaconf import DictConfig
import lightning as L
from pathlib import Path

import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Evaluate ESM2-only baseline: validate to optimize threshold, then test."""
    
    # Set seed
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)
    
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)
    
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    log.info("Instantiating loggers...")
    logger = []
    if cfg.get("logger"):
        from src.utils import instantiate_loggers
        logger = instantiate_loggers(cfg.get("logger"))
    
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger)
    
    # Setup datamodule to create datasets
    log.info("Setting up datamodule...")
    datamodule.setup(stage="fit")  # Creates train and val datasets
    datamodule.setup(stage="test")  # Creates test dataset
    
    # Step 1: Run validation to find optimal threshold
    log.info("="*60)
    log.info("Step 1: Running validation to find optimal threshold...")
    log.info("="*60)
    trainer.validate(model=model, datamodule=datamodule)
    
    log.info(f"\n{'='*60}")
    log.info(f"Validation complete! Optimal threshold: {model.pred_threshold:.4f}")
    log.info(f"{'='*60}\n")
    
    # Step 2: Save checkpoint with optimal threshold
    save_dir = Path(cfg.paths.output_dir) / "checkpoints"
    save_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = save_dir / f"esm2_only_thresh{model.pred_threshold:.3f}.ckpt"
    
    log.info(f"Saving checkpoint to: {ckpt_path}")
    trainer.save_checkpoint(ckpt_path)
    
    # Step 3: Run test with optimal threshold
    log.info("="*60)
    log.info("Step 2: Running test with optimal threshold...")
    log.info("="*60)
    trainer.test(model=model, datamodule=datamodule)
    
    log.info(f"\n{'='*60}")
    log.info(f"Evaluation complete!")
    log.info(f"Checkpoint saved: {ckpt_path}")
    log.info(f"{'='*60}\n")
    
    return trainer.callback_metrics


if __name__ == "__main__":
    main()
