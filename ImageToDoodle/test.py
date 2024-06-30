import torch
from ImageToDoodle.utils import save_checkpoint, load_checkpoint, save_some_examples,save_test_examples
import torch.nn as nn
import torch.optim as optim
import ImageToDoodle.config as config
from ImageToDoodle.dataset import TestDataset
from ImageToDoodle.generator_model import Generator
from ImageToDoodle.discriminator_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.utils import save_image
import numpy as np

torch.backends.cudnn.benchmark = True




def main(num_residuals,image_bytes):


    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(img_channels=3, num_features=64,num_residuals=num_residuals).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    GEN_DIR="static/20gen.pth.tar"
    DISC_DIR="static/20disc.pth.tar"

    load_checkpoint(
        GEN_DIR, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        DISC_DIR, disc, opt_disc, config.LEARNING_RATE,
    )

    #print(gen)

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    test_dataset = TestDataset(root_dir=config.TEST_DIR)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)


    for epoch in range(0,1):
        save_test_examples(gen,disc, test_loader, epoch, folder="static/images/")


