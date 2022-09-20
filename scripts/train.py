import os
import matplotlib.pyplot as plt
import pandas as pd
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
import numpy as np
import torch
import cv2
from PIL import Image
from omuti.torch.main import OmutiModule

TRAINING = True


def predict(df, savedir, ground_truth=None):
    os.makedirs(savedir, exist_ok=True)
    for name, _df in df.groupby("image_path"):
        basename = os.path.splitext(os.path.basename(name))[0]
        image = np.array(Image.open(name).convert('RGB')).copy()
        for _, row in _df.iterrows():
            image = cv2.rectangle(
                image,
                (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                color=(255, 0, 0),
                thickness=1,
                lineType=cv2.LINE_AA
            )

        if ground_truth is not None:
            annotations = ground_truth[ground_truth.image_path == name]
            for _, row in annotations.iterrows():
                image = cv2.rectangle(
                    image,
                    (int(row["xmin"]), int(row["ymin"])), (int(row["xmax"]), int(row["ymax"])),
                    color=(0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )

        cv2.imwrite(f'{savedir}/{basename}.png', image)


if __name__ == '__main__':

    # TODO: Support multiple features
    feature_name = 'Omuti1972'

    if TRAINING:

        training_file = f'scratch/ml/{feature_name}/training.csv'
        validation_file = f'scratch/ml/{feature_name}/validation.csv'

        model = OmutiModule(training_file=training_file, validation_file=validation_file)

        model.model.load_state_dict(
            torch.load('neon.pt', map_location=model.device))

        model.label_dict = {feature_name: 0}
        model.create_trainer(logger=TensorBoardLogger(save_dir='lightning_logs/'), log_every_n_steps=1)
        model.trainer.fit(model)

        torch.save(model.state_dict(), 'model.pth')

    else:

        model = OmutiModule()

        model.load_state_dict(torch.load('model.pth'))
        model.eval()  # set dropout and batch normalization layers to evaluation mode before running inference

        # Images from test set
        # testing_file = "testing.csv"
        # ground_truth = pd.read_csv(testing_file)
        # predictions = model.predict_file(testing_file, root_dir=os.path.dirname(testing_file))
        # predict(predictions, savedir='predictions', ground_truth=ground_truth)

        # A large hitherto unseen raster tif
        raster_path = 'scratch/subset_random_reprojected.tif'
        # Window size of 300px with an overlap of 5% among windows
        df = model.predict_tile(raster_path, return_plot=False, patch_size=300, patch_overlap=0.05, thickness=1)
        df.to_csv('predictions/tile.csv', index=False)
        im = model.predict_tile(raster_path, return_plot=True, patch_size=300, patch_overlap=0.05, thickness=1)
        cv2.imwrite('predictions/tile.png', im)

        # Preprocess
        # output_annotations = split_raster(
        #     path_to_raster='scratch/subset_random.tif',
        #     annotations_file='scratch/annotations.csv',
        #     base_dir='crops',
        #     patch_size=300
        # )
        #
        # plt.imshow(im)
        #
        # figure = plt.gcf()
        # figure.set_size_inches(8, 6)
        # plt.savefig('output2.png')
