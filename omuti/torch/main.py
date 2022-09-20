import os
import pandas as pd
from PIL import Image
import torch

import pytorch_lightning as pl
from torch import optim
import numpy as np

from omuti.torch import utilities
from omuti.torch import dataset
from omuti.torch import model
from omuti.torch import predict
from omuti.torch import evaluate as evaluate_iou
from pytorch_lightning.callbacks import LearningRateMonitor


class OmutiModule(pl.LightningModule):
    def __init__(self, label_dict={"Omuti": 0}, transforms=None, training_file=None, validation_file=None):
        super().__init__()

        # Pytorch lightning handles the device, but we need one for adhoc methods like predict_image.
        if torch.cuda.is_available():
            self.current_device = torch.device("cuda")
        else:
            self.current_device = torch.device("cpu")

        self.training_file = training_file
        self.validation_file = validation_file

        self.num_classes = len(label_dict)
        self.create_model()

        self.label_dict = label_dict
        self.numeric_to_label_dict = {v: k for k, v in label_dict.items()}

        # Add user supplied transforms
        if transforms is None:
            self.transforms = dataset.get_transform
        else:
            self.transforms = transforms

        self.save_hyperparameters()

    def create_model(self):
        self.model = model.create_model(self.num_classes, 0.05, 0.1)  # TODO

    def create_trainer(self, logger=None, callbacks=[], **kwargs):

        lr_monitor = LearningRateMonitor(logging_interval='epoch')
        callbacks.append(lr_monitor)

        self.trainer = pl.Trainer(logger=logger,
                                  max_epochs=1,  # TODO
                                  gpus=0,  # TODO
                                  enable_checkpointing=False,
                                  accelerator='gpu',
                                  fast_dev_run=False,
                                  callbacks=callbacks,
                                  **kwargs)

    def save_model(self, path):
        self.trainer.save_checkpoint(path)

    def load_dataset(self,
                     csv_file,
                     augment=False,
                     shuffle=True,
                     batch_size=1,
                     train=False):

        ds = dataset.TreeDataset(csv_file=csv_file,
                                 transforms=self.transforms(augment=augment),
                                 label_dict=self.label_dict,
                                 preload_images=False)

        data_loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=utilities.collate_fn,
            num_workers=6,
        )

        return data_loader

    def train_dataloader(self):
        loader = self.load_dataset(csv_file=self.training_file,
                                   augment=True,
                                   shuffle=True,
                                   batch_size=1)

        return loader

    def val_dataloader(self):
        loader = self.load_dataset(csv_file=self.validation_file,
                                   augment=False,
                                   shuffle=False,
                                   batch_size=1)

        return loader

    def predict_image(self, image=None, path=None, return_plot=False, color=None, thickness=1):
        if isinstance(image, str):
            raise ValueError(
                "Path provided instead of image. If you want to predict an image from disk, is path ="
            )

        if path:
            if not isinstance(path, str):
                raise ValueError("Path expects a string path to image on disk")
            image = np.array(Image.open(path).convert("RGB")).astype("float32")

        # sanity checks on input images
        if not type(image) == np.ndarray:
            raise TypeError(
                "Input image is of type {}, expected numpy, if reading from PIL, wrap in np.array(image).astype(float32)".format(
                    type(image)))

            # Load on GPU is available
        if self.current_device.type == "cuda":
            self.model = self.model.to("cuda")

        self.model.eval()
        self.model.score_thresh = 0.1

        # Check if GPU is available and pass image to gpu
        result = predict.predict_image(model=self.model,
                                       image=image,
                                       return_plot=return_plot,
                                       device=self.current_device,
                                       iou_threshold=0.05,  # TODO
                                       color=color,
                                       thickness=thickness)

        # Set labels to character from numeric if returning boxes df
        if not return_plot:
            if not result is None:
                result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])

        return result

    def predict_file(self, csv_file, root_dir, savedir=None, color=None, thickness=1):
        """Create a dataset and predict entire annotation file

        Csv file format is .csv file with the columns "image_path", "xmin","ymin","xmax","ymax" for the image name and bounding box position.
        Image_path is the relative filename, not absolute path, which is in the root_dir directory. One bounding box per line.

        Args:
            csv_file: path to csv file
            root_dir: directory of images. If none, uses "image_dir" in config
            savedir: Optional. Directory to save image plots.
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px
        Returns:
            df: pandas dataframe with bounding boxes, label and scores for each image in the csv file
        """
        self.model = self.model.to(self.current_device)
        self.model.eval()
        self.model.score_thresh = 0.1

        result = predict.predict_file(model=self.model,
                                      csv_file=csv_file,
                                      root_dir=root_dir,
                                      savedir=savedir,
                                      device=self.current_device,
                                      iou_threshold=0.05,  # TODO
                                      color=color,
                                      thickness=thickness)

        # Set labels to character from numeric
        result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])

        return result

    def predict_tile(self,
                     raster_path=None,
                     image=None,
                     patch_size=400,
                     patch_overlap=0.05,
                     iou_threshold=0.15,
                     return_plot=False,
                     mosaic=True,
                     use_soft_nms=False,
                     sigma=0.5,
                     thresh=0.001,
                     color=None,
                     thickness=1):
        """For images too large to input into the model, predict_tile cuts the
        image into overlapping windows, predicts trees on each window and
        reassambles into a single array.

        Args:
            raster_path: Path to image on disk
            image (array): Numpy image array in BGR channel order
                following openCV convention
            patch_size: patch size default400,
            patch_overlap: patch overlap default 0.15,
            iou_threshold: Minimum iou overlap among predictions between
                windows to be suppressed. Defaults to 0.5.
                Lower values suppress more boxes at edges.
            return_plot: Should the image be returned with the predictions drawn?
            mosaic: Return a single prediction dataframe (True) or a tuple of image crops and predictions (False)
            use_soft_nms: whether to perform Gaussian Soft NMS or not, if false, default perform NMS.
            sigma: variance of Gaussian function used in Gaussian Soft NMS
            thresh: the score thresh used to filter bboxes after soft-nms performed
            color: color of the bounding box as a tuple of BGR color, e.g. orange annotations is (0, 165, 255)
            thickness: thickness of the rectangle border line in px

        Returns:
            boxes (array): if return_plot, an image.
            Otherwise a numpy array of predicted bounding boxes, scores and labels
        """

        # Load on GPU is available
        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.model.eval()
        self.model.score_thresh = 0.1  # TODO
        self.model.nms_thresh = 0.05  # TODO

        result = predict.predict_tile(model=self.model,
                                      raster_path=raster_path,
                                      image=image,
                                      patch_size=patch_size,
                                      patch_overlap=patch_overlap,
                                      iou_threshold=iou_threshold,
                                      return_plot=return_plot,
                                      mosaic=mosaic,
                                      use_soft_nms=use_soft_nms,
                                      sigma=sigma,
                                      thresh=thresh,
                                      device=self.current_device,
                                      color=color,
                                      thickness=thickness)

        # edge case, if no boxes predictioned return None
        if result is None:
            print("No predictions made, returning None")
            return None

        # Set labels to character from numeric if returning boxes df
        if not return_plot:
            if mosaic:
                result["label"] = result.label.apply(lambda x: self.numeric_to_label_dict[x])
            else:
                for df, image in result:
                    df["label"] = df.label.apply(lambda x: self.numeric_to_label_dict[x])

        return result

    def training_step(self, batch, batch_idx):
        """Train on a loaded dataset
        """
        # Confirm model is in train mode
        self.model.train()

        # allow for empty data if data augmentation is generated
        path, images, targets = batch

        loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        return losses

    def validation_step(self, batch, batch_idx):
        """Train on a loaded dataset

        """
        try:
            path, images, targets = batch
        except:
            print("Empty batch encountered, skipping")
            return None

        self.model.train()
        with torch.no_grad():
            loss_dict = self.model.forward(images, targets)

        # sum of regression and classification loss
        losses = sum([loss for loss in loss_dict.values()])

        # Log loss
        for key, value in loss_dict.items():
            self.log("val_{}".format(key), value, on_epoch=True)

        return losses

    def on_epoch_end(self):
        if self.validation_file is not None:
            results = self.evaluate(csv_file=self.validation_file)
            self.log("box_recall", results["box_recall"])
            self.log("box_precision", results["box_precision"])

            if not type(results["class_recall"]) == type(None):
                for index, row in results["class_recall"].iterrows():
                    self.log("{}_Recall".format(self.numeric_to_label_dict[row["label"]]), row["recall"])
                    self.log("{}_Precision".format(self.numeric_to_label_dict[row["label"]]), row["precision"])

    def configure_optimizers(self):
        optimizer = optim.SGD(self.model.parameters(),
                              lr=0.001,
                              momentum=0.9)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=0.1,
                                                               patience=10,
                                                               verbose=True,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

        return {'optimizer': optimizer, 'lr_scheduler': scheduler, "monitor": 'val_classification'}

    def evaluate(self,
                 csv_file,
                 iou_threshold=None,
                 savedir=None):

        if torch.cuda.is_available():
            self.model = self.model.to("cuda")

        self.model.eval()
        self.model.score_thresh = 0.1

        predictions = predict.predict_file(model=self.model,
                                           csv_file=csv_file,
                                           savedir=savedir,
                                           device=self.current_device,
                                           iou_threshold=0.05)

        ground_df = pd.read_csv(csv_file)
        ground_df["label"] = ground_df.label.apply(lambda x: self.label_dict[x])

        # remove empty samples from ground truth
        ground_df = ground_df[~((ground_df.xmin == 0) & (ground_df.xmax == 0))]

        # if no arg for iou_threshold, set as config
        if iou_threshold is None:
            iou_threshold = 0.4

        results = evaluate_iou.evaluate(predictions=predictions,
                                        ground_df=ground_df,
                                        iou_threshold=iou_threshold,
                                        savedir=savedir)

        # replace classes if not NUll, wrap in try catch if no predictions
        if not results["results"].empty:
            results["results"]["predicted_label"] = results["results"]["predicted_label"].apply(
                lambda x: self.numeric_to_label_dict[x] if not pd.isnull(x) else x)
            results["results"]["true_label"] = results["results"]["true_label"].apply(
                lambda x: self.numeric_to_label_dict[x])
            results["predictions"] = predictions

        return results
