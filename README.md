# Art Style Classification

Classifying paintings across 8 artistic styles using supervised, self-supervised, and generative approaches.

## Dataset

1422 images across 8 classes: ArtDeco, Cubism, Impressionism, Japonism, Naturalism, Rococo, cartoon, photo.  
Split: 851 train / 285 val / 286 test.

## Project Structure

```
notebooks/
    1_EDA.ipynb                 # Dataset exploration and class distribution
    2_Classifier_Training.ipynb # ResNet18 and EfficientNet B0 fine-tuning
    3_SSL_Training.ipynb        # SimCLR pretraining, linear probing, fine-tuning
    4_XAI.ipynb                 # Grad-CAM++ explainability for all models
    5_VAE_Clustering.ipynb      # VAE training, clustering, XGBoost classification
    6_Results.ipynb             # Convergence plots and model comparison

src/iml2_painting_style/
    model_resnet.py             # ResNet18 classifier
    model_efficientnet.py       # EfficientNet B0 classifier
    simclr_model.py             # SimCLR backbone and projection head
    simclr_engine.py            # SimCLR training and evaluation loops
    simclr_loss.py              # NT-Xent loss
    simclr_transforms.py        # Contrastive augmentation pipeline
    vae.py                      # Convolutional VAE and ELBO loss
    vae_engine.py               # VAE training, evaluation, embedding extraction
    dataset.py                  # Dataset loader
    transforms.py               # Supervised augmentation pipeline
    engine.py                   # Supervised training and evaluation loops
    metrics.py                  # Accuracy and macro F1
    checkpoints.py              # Save and load checkpoints
    utils.py                    # Seed and device utilities

checkpoints/                    # Saved model weights (not tracked in git, see below)
reports/                        # Saved plots and model comparison CSV
```

## Setup

```bash
# create virtual environment
python -m venv venv

# activate it
source venv/bin/activate        # on Windows: venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

## Running the Notebooks

Run the notebooks in order:

1. `1_EDA.ipynb` : explore the dataset
2. `2_Classifier_Training.ipynb` : train ResNet18 and EfficientNet B0
3. `3_SSL_Training.ipynb` : train SimCLR (3 stages)
4. `5_VAE_Clustering.ipynb` :train VAE, cluster embeddings, train XGBoost
5. `4_XAI.ipynb` : generate Grad-CAM++ visualizations (requires trained models)
6. `6_Results.ipynb` : plot convergence curves and compare all models

## Checkpoints

Pretrained model checkpoints are not tracked in git. You can download them from Google Drive and place them in the `checkpoints/` directory:

[Download checkpoints](https://drive.google.com/drive/folders/1djLYn4DfqV1F_T-phAzwpIs2Lh0aDNho?usp=sharing)

## Results

| Model | Accuracy | Macro F1 |
|---|---|---|
| ResNet18 (supervised) | 76.2% | 0.716 |
| EfficientNet B0 (supervised) | 74.5% | 0.711 |
| SimCLR + Fine-tuning (SSL) | 75.5% | 0.720 |
| VAE embeddings + XGBoost | 35% | 0.308 |

