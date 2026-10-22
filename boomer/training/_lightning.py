"""Shared Lightning imports with the existing legacy namespace fallback."""

try:  # Lightning >= 2.0 namespace, with legacy fallback.
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from lightning.pytorch.loggers import WandbLogger
except ImportError:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        Callback,
        LearningRateMonitor,
        ModelCheckpoint,
        TQDMProgressBar,
    )
    from pytorch_lightning.loggers import WandbLogger
