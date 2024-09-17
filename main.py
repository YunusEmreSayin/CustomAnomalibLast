from lightning.pytorch.trainer import Trainer

from pathlib import Path
import yaml
import numpy as np
from anomalib.metrics import ManualThreshold,F1AdaptiveThreshold
import torch
from PIL import Image
from anomalib.data.utils import read_image,TestSplitMode
from anomalib.deploy import ExportType, OpenVINOInferencer
from utils.engine import Engine
from matplotlib import pyplot as plt
from torchvision.transforms.v2 import Resize
from torchvision.transforms import ToPILImage
from pathlib import Path
from anomalib.data.image.mvtec import MVTec, MVTecDataset
from anomalib.models.image.patchcore import Patchcore
from anomalib.models.image.patchcore.anomaly_map import AnomalyMapGenerator
from anomalib import TaskType
from anomalib.data import Folder
import os
from anomalib.data.image import folder
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms
from anomalib.data.image.folder import Folder, FolderDataset
import yaml
from anomalib.deploy import OpenVINOInferencer
import cv2
from torchvision.transforms.functional import to_pil_image
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from anomalib.loggers import AnomalibWandbLogger
from anomalib.deploy import ExportType
from anomalib.deploy import TorchInferencer
from torchvision.transforms.v2.functional import to_dtype, to_image
from torch import as_tensor
import time
from skimage import measure
from anomalib.callbacks.checkpoint import ModelCheckpoint
from anomalib.callbacks.metrics import _MetricsCallback
from anomalib.callbacks.normalization import get_normalization_callback
from anomalib.callbacks.normalization.base import NormalizationCallback
from anomalib.callbacks.post_processor import _PostProcessorCallback
from anomalib.callbacks.thresholding import _ThresholdCallback
from anomalib.callbacks.timer import TimerCallback
from anomalib.callbacks.visualizer import _VisualizationCallback
from utils.patchcore.lightning_model import Patchcore
import json
import logging
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import nn
from torchvision.transforms.v2 import Transform
from anomalib.models import AnomalyModule
from anomalib import TaskType
from anomalib.data import AnomalibDataModule
from anomalib.deploy.export import CompressionType, ExportType, InferenceModel
from anomalib.utils.exceptions import try_import
from lightning.pytorch import Callback
from anomalib.deploy import CompressionType, ExportType
from lightning.pytorch.utilities.types import _EVALUATE_OUTPUT, _PREDICT_OUTPUT, EVAL_DATALOADERS, TRAIN_DATALOADERS
from anomalib.models.components.base.export_mixin import ExportMixin
from anomalib import LearningType, TaskType
if TYPE_CHECKING:
    from torch.types import Number
from lightning.pytorch.callbacks import Callback, RichModelSummary, RichProgressBar

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    config_path = Path("config.yaml")# The path of config file on Google Drive
    print("Reading Config File....")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    """ Reading Config.yaml """
    print(config)# Checking config content

    root_path= Path(config["dataset"]["path"])# Defining root path from config

    """Defining Dataset and DataLoaders in here"""
    dataModule=Folder(
        name=config["dataset"]["name"],
        root=root_path,
        task=TaskType.CLASSIFICATION,
        test_split_mode=TestSplitMode.NONE,
        normal_dir=config["dataset"]["normal_dir"],
        abnormal_dir=config["dataset"]["abnormal_dir"],
        image_size=(224,224),
        num_workers=config["dataset"]["num_workers"],
    )
    """
        Args:
          name--> The name for our dataset
          root--> The path of the root directory of dataset
          task--> Task type for our operations
              ---Values:CLASSIFICATION,SEGMENTATION,DETECTION
          test_split_mode-->Seperating mode for test set
          normal_dir--> Path of the good images
          abnormal_dir--> Path of the defective images
          image_size--> Definition of the resolution of images we have
          num_workers--> Count of threads we will use
    
    """

    dataModule.setup()# Defines the datamodule
    """Defining the dataloaders"""
    train_loader = dataModule.train_dataloader()
    #test_loader = dataModule.test_dataloader()


    """Setting up the Train,Test and Validation datasets from DataLoader"""
    #train
    i, data_train = next(enumerate(dataModule.train_dataloader()))
    print(data_train.keys(), data_train["image"].shape)
    img_train = to_pil_image(data_train["image"][0].clone())

    #validation
    #i, data_val = next(enumerate(dataModule.val_dataloader()))
    #img_val = to_pil_image(data_val["image"][0].clone())
    ##test
    #i, data_test = next(enumerate(dataModule.test_dataloader()))
    #img_test = to_pil_image(data_test["image"][0].clone())

    train_dataset = dataModule.train_data.samples
    #test_dataset = dataModule.test_data.samples
    #val_dataset = dataModule.val_data.samples

    """Log Commands"""
    print("TRAIN DATASET FEATURES")
    print(train_dataset.info())
    print("")
    print("IMAGE DISTRIBUTION BY CLASS")
    print("")
    desc_grouped = train_dataset[['label']].value_counts()
    print(desc_grouped)
    # print("----------------------------------------------------------")
    # print("TEST DATASET FEATURES")
    # print(test_dataset.info())
    # print("")
    # print("IMAGE DISTRIBUTION BY CLASS")
    # print("")
    # desc_grouped = test_dataset[['label']].value_counts()
    # print(desc_grouped)
    # print("----------------------------------------------------------")
    # print("VAL DATASET FEATURES")
    # print(val_dataset.info())
    # print("")
    # print("IMAGE DISTRIBUTION BY CLASS")
    # print("")
    # desc_grouped = val_dataset[['label']].value_counts()
    # print(desc_grouped)


    """ Managing the model.pt File"""
    class ExportMixin(ExportMixin):
      def _create_export_root(self,export_root: str | Path, export_type: ExportType) -> Path:
          """Create export directory.

          Args:
              export_root (str | Path): Path to the root folder of the exported model.
              export_type (ExportType): Mode to export the model. Torch, ONNX or OpenVINO.

          Returns:
              Path: Path to the export directory.
          """
          export_root = Path(export_root)
          export_root.mkdir(parents=True, exist_ok=True)
          return export_root


      def to_torch(
            self,
            export_root: Path | str,
            transform: Transform | None = None,
            task: TaskType | None = None,
      ) -> Path:
            """Export AnomalibModel to torch.

            Args:
                export_root (Path): Path to the output folder.
                transform (Transform, optional): Input transforms used for the model. If not provided, the transform is
                    taken from the model.
                    Defaults to ``None``.
                task (TaskType | None): Task type.
                    Defaults to ``None``.

            Returns:
                Path: Path to the exported pytorch model.

            Examples:
                Assume that we have a model to train and we want to export it to torch format.

                --> from anomalib.data import Visa
                --> from anomalib.models import Patchcore
                --> from anomalib.engine import Engine
                ...
                --> datamodule = Visa()
                --> model = Patchcore()
                --> engine = Engine()
                ...
                --> engine.fit(model, datamodule)

                Now that we have a model trained, we can export it to torch format.

                --> model.to_torch(
                ...     export_root="path/to/export",
                ...     transform=datamodule.test_data.transform,
                ...     task=datamodule.test_data.task,
                ... )
            """
            transform = transform or self.transform or self.configure_transforms()
            inference_model = InferenceModel(model=self.model, transform=transform)
            export_root = self._create_export_root(export_root, ExportType.TORCH)
            metadata = self._get_metadata(task=task)
            pt_model_path = export_root / "serkon_torch.pt"
            torch.save(
                obj={"model": inference_model, "metadata": metadata},
                f=pt_model_path,
            )
            return pt_model_path


    class AnomalyModule(AnomalyModule,ExportMixin):
      pass
    class Patchcore(Patchcore,AnomalyModule):
      pass
    class UnassignedError(Exception):
        """Unassigned error."""


    """Defining the custom engine"""

    class SerkonAnomalyEngine(Engine):



        def fit(
            self,
            model: AnomalyModule,
            train_dataloaders: TRAIN_DATALOADERS | None = None,
            val_dataloaders: EVAL_DATALOADERS | None = None,
            datamodule: AnomalibDataModule | None = None,
            ckpt_path: str | Path | None = None,
        ) -> None:
            """Fit the model using the trainer.

            Args:
                model (AnomalyModule): Model to be trained.
                train_dataloaders (TRAIN_DATALOADERS | None, optional): Train dataloaders.
                    Defaults to None.
                val_dataloaders (EVAL_DATALOADERS | None, optional): Validation dataloaders.
                    Defaults to None.
                datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                    If provided, dataloaders will be instantiated from this.
                    Defaults to None.
                ckpt_path (str | None, optional): Checkpoint path. If provided, the model will be loaded from this path.
                    Defaults to None.

            CLI Usage:
                1. you can pick a model, and you can run through the MVTec dataset.
                    ```python
                    anomalib fit --model anomalib.models.Padim
                    ```
                2. Of course, you can override the various values with commands.
                    ```python
                    anomalib fit --model anomalib.models.Padim --data <CONFIG | CLASS_PATH_OR_NAME> --trainer.max_epochs 3
                    ```
                4. If you have a ready configuration file, run it like this.
                    ```python
                    anomalib fit --config <config_file_path>
                    ```
            """
            if ckpt_path:
                ckpt_path = Path(ckpt_path).resolve()

            self._setup_workspace(
                model=model,
                train_dataloaders=train_dataloaders,
                val_dataloaders=val_dataloaders,
                datamodule=datamodule,
                versioned_dir=True,
            )
            self._setup_trainer(model)
            self._setup_dataset_task(train_dataloaders, val_dataloaders, datamodule)
            self._setup_transform(model, datamodule=datamodule, ckpt_path=ckpt_path)
            if model.learning_type in [LearningType.ZERO_SHOT, LearningType.FEW_SHOT]:
                # if the model is zero-shot or few-shot, we only need to run validate for normalization and thresholding
                self.trainer.validate(model, val_dataloaders, datamodule=datamodule, ckpt_path=ckpt_path)
            else:
                self.trainer.fit(model, train_dataloaders, val_dataloaders, datamodule, ckpt_path)
            print(f"[LOG] Learning Type is ---> {model.learning_type}")

        @property
        def trainer(self) -> Trainer:
            """Property to get the trainer.

            Raises:
                UnassignedError: When the trainer is not assigned yet.

            Returns:
                Trainer: Lightning Trainer.
            """
            if not self._trainer:
                msg = "``self.trainer`` is not assigned yet."
                raise UnassignedError(msg)
            # self._trainer.enable_model_summary = False
            # self._trainer.enable_progress_bar = False
            return self._trainer
        def _setup_trainer(self, model: AnomalyModule) -> None:
            """Instantiate the trainer based on the model parameters."""
            # Check if the cache requires an update
            print(f"[LOG] Cache args --> {self._cache.args}")
            if self._cache.requires_update(model):
                self._cache.update(model)

            # Setup anomalib callbacks to be used with the trainer
            self._setup_anomalib_callbacks()

            # Temporarily set devices to 1 to avoid issues with multiple processes
            self._cache.args["devices"] = 1


            # Instantiate the trainer if it is not already instantiated
            if self._trainer is None:
                print("[LOG] Setting Model Trainer...")
                self._trainer = Trainer(**self._cache.args)
                self._trainer.enable_model_summary = False
                self._trainer.enable_progress_bar = False
                self._trainer.enable_checkpointing = False
            print(f"[LOG] Trainer Callback args --> {self._trainer.callbacks}")
            print(f"[LOG] Trainer Args logs --> {type(self._trainer)}")
            print(f"[LOG] Model Summary State --> {self._trainer.enable_model_summary}")
            print(f"[LOG] Progress Bar State --> {self._trainer.enable_progress_bar}")
            print(f"[LOG] Chekpointing State --> {self._trainer.enable_checkpointing}")




      ## Disabled the Checkpoint callback for blocking the creation of ckpt
        @property
        def checkpoint_callback(self) -> None :
            """The ``ModelCheckpoint`` callback in the trainer.callbacks list, or ``None`` if it doesn't exist.

            Returns:;
                ModelCheckpoint | None: ModelCheckpoint callback, if available.
            """
            if self._trainer is None:
                return None
            return None

        @property
        def best_model_path(self) -> None :
            """The path to the best model checkpoint.

            Returns:
                str: Path to the best model checkpoint.
            """
            if self.checkpoint_callback is None:
                return None
            return None


        def _setup_anomalib_callbacks(self) -> None:
            """Set up callbacks for the trainer."""
            _callbacks: list[Callback] = []

            # Add the post-processor callbacks.
            _callbacks.append(_PostProcessorCallback())

            # Add the the normalization callback.
            normalization_callback = get_normalization_callback(self.normalization)
            if normalization_callback is not None:
                _callbacks.append(normalization_callback)

            # Add the thresholding and metrics callbacks.
            _callbacks.append(_ThresholdCallback(self.threshold))
            _callbacks.append(_MetricsCallback(self.task, self.image_metric_names, self.pixel_metric_names))



            _callbacks.append(TimerCallback())
            print(f"[LOG] Callback info--> Length of Callbacks:{len(_callbacks)} | Type of Callbacks:{type(_callbacks)} ")
            for cb in _callbacks:
              print(f"[LOG] Callback info--> Callback Name: {cb} |  Callback State: {hasattr(cb,'state_dict')} | Instance State:{isinstance(cb,Callback)} | Type Of Callback: {type(cb)}")

            # Combine the callbacks, and update the trainer callbacks.
            self._cache.args["callbacks"] = _callbacks + self._cache.args["callbacks"]
        def export(
            self,
            model: AnomalyModule,
            export_type: ExportType | str,
            export_root: str | Path | None = None,
            input_size: tuple[int, int] | None = None,
            transform: Transform | None = None,
            compression_type: CompressionType | None = None,
            datamodule: AnomalibDataModule | None = None,
            ov_args: dict[str, Any] | None = None,
            ckpt_path: str | Path | None = None,
        ) -> Path | None:


          """Export the model in PyTorch, ONNX or OpenVINO format.

            Args:
                model (AnomalyModule): Trained model.
                export_type (ExportType): Export type.
                export_root (str | Path | None, optional): Path to the output directory. If it is not set, the model is
                    exported to trainer.default_root_dir. Defaults to None.
                input_size (tuple[int, int] | None, optional): A statis input shape for the model, which is exported to ONNX
                    and OpenVINO format. Defaults to None.
                transform (Transform | None, optional): Input transform to include in the exported model. If not provided,
                    the engine will try to use the default transform from the model.
                    Defaults to ``None``.
                compression_type (CompressionType | None, optional): Compression type for OpenVINO exporting only.
                    Defaults to ``None``.
                datamodule (AnomalibDataModule | None, optional): Lightning datamodule.
                    Must be provided if CompressionType.INT8_PTQ is selected.
                    Defaults to ``None``.
                ov_args (dict[str, Any] | None, optional): This is optional and used only for OpenVINO's model optimizer.
                    Defaults to None.
                ckpt_path (str | Path | None): Checkpoint path. If provided, the model will be loaded from this path.

            Returns:
                Path: Path to the exported model.

            Raises:
                ValueError: If Dataset, Datamodule, and transform are not provided.
                TypeError: If path to the transform file is not a string or Path.

            CLI Usage:
                1. To export as a torch ``.pt`` file you can run the following command.
                    ```python
                    anomalib export --model Padim --export_mode torch --ckpt_path <PATH_TO_CHECKPOINT>
                    ```
                2. To export as an ONNX ``.onnx`` file you can run the following command.
                    ```python
                    anomalib export --model Padim --export_mode onnx --ckpt_path <PATH_TO_CHECKPOINT> \
                    --input_size "[256,256]"
                    ```
                3. To export as an OpenVINO ``.xml`` and ``.bin`` file you can run the following command.
                    ```python
                    anomalib export --model Padim --export_mode openvino --ckpt_path <PATH_TO_CHECKPOINT> \
                    --input_size "[256,256]"
                    ```
                4. You can also override OpenVINO model optimizer by adding the ``--ov_args.<key>`` arguments.
                    ```python
                    anomalib export --model Padim --export_mode openvino --ckpt_path <PATH_TO_CHECKPOINT> \
                    --input_size "[256,256]" --ov_args.compress_to_fp16 False
                    ```
          """
          export_type = ExportType(export_type)
          self._setup_trainer(model)
          if ckpt_path:
              ckpt_path = Path(ckpt_path).resolve()
              model = model.__class__.load_from_checkpoint(ckpt_path)

          if export_root is None:
              export_root = Path(self.trainer.default_root_dir)

          exported_model_path: Path | None = None
          if export_type == ExportType.TORCH:
              exported_model_path = model.to_torch(
                  export_root=export_root,
                  transform=transform,
                  task=self.task,
          )

          else:
              logging.error(f"Export type {export_type} is not supported yet.")

          if exported_model_path:
              logging.info(f"Exported model to {exported_model_path}")
          return exported_model_path

    """Defining the model"""
    model = Patchcore(
        backbone=config["model"]["backbone"],
        pre_trained=config["model"]["pre_trained"],
        coreset_sampling_ratio=config["model"]["coreset_sampling_ratio"],
        num_neighbors=config["model"]["num_neighbors"],
        layers=config["model"]["layers"],
    )
    """
    Args:
      backbone--> The Neural Network for our model
      pre_trained--> Status of the model is pre_trained or not
      coreset_sampling_ratio--> This parameters controls how much data will be kept on memory
        This parameter effect the accuracy and speed of model
      num_neighbors-->this param controls the models closeness and kinship relationships
    
    """
    #model.state_dict()

    callbacks = [
        # ModelCheckpoint(
        #     mode="max",
        #     monitor="image_AUROC",
        #     save_last=True,
        #     verbose=True,
        #     auto_insert_metric_name=True,
        #     every_n_epochs=1,
        # ),
        EarlyStopping(
            monitor="image_AUROC",
            mode="max",
            patience=10,
        ),
    ]

    #wandb üzerinden loglar görüntünlenebilir API key en üst kısımda text içinde var
    # wandb_logger = AnomalibWandbLogger(project="image_anomaly_detection",
    #             name="name_wandb_experiment")


    """Setting the optimal threshold value manually"""
    manual_threshold=ManualThreshold(default_value=16.5)#19.5
    print(f"Threshold Type: {type(manual_threshold)} Threshold Value: {manual_threshold.compute()}")

    """Setting up the engine"""
    engine = SerkonAnomalyEngine(
        max_epochs=200,
        #callbacks=callbacks,
        pixel_metrics="AUPRO",
        accelerator="cpu",  # \<"cpu", "gpu", "tpu", "ipu", "hpu", "auto">,
        devices=1,
        # logger=wandb_logger,
        task=TaskType.CLASSIFICATION,
        threshold =manual_threshold,
        enable_model_summary = False,
        enable_progress_bar = False,
        enable_checkpointing = False)
    """
    Args:
      max_epochs--> The number of epochs we will train
      callbacks--> The callbacks we will use
      pixel_metrics-->
      accelerator--> Device selection for engine
      devices--> How many cpu's we will use
      logger--> The logging method we have
        wandb--> This is optional, wandb is a common logging service
      task--> Task type of the operation that engine we will run
      threshold--> The threshold value for inspection
    """

    """Training the model"""

    trainStart = time.time()
    # engine._cache.args['callbacks'] = []
    print("Fit...")
    engine.fit(datamodule=dataModule, model=model)
    trainEnd = time.time()
    train_duration = (trainEnd - trainStart) * 1000
    print(f"Train duration: {train_duration:.2f} ms")#train finished and we printed the train duration

    """Testing the model"""
    #print("Test...")
    #testStart = time.time()
    #engine.test(datamodule=dataModule, model=model)
    #testEnd = time.time()
    #test_duration= (testEnd - testStart) * 1000
    #print(f"Test duration: {test_duration:.2f} ms")# testing finished and we printed the test duration

    ## Exporting features/weights we learned from engine for prediction
    print("Export weights...")
    # path_export_weights = engine.export(export_type=ExportType.TORCH,
    #                                     model=model,
    #                                     )
    # print("path_export_weights: ", path_export_weights)


    print(engine.__dict__)
    #engine.log_train_params()
    print(engine.checkpoint_callback)
    print(engine.trainer.enable_model_summary)
    #print(engine._cache.args['callbacks'].clear())
    print(engine._cache.args['enable_model_summary'])
    print(engine._cache.args)