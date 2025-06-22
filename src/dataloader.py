from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageFolderWithPaths(datasets.ImageFolder):
    def __getitem__(self, index):
        original_tuple = super().__getitem__(index)
        path = self.imgs[index][0]
        return original_tuple + (path,)  # (image_tensor, label, path)

def get_dataloaders(data_dir, batch_size = 64):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3)
    ])

    train_data = ImageFolderWithPaths(root=f"{data_dir}\\train", transform=transform)
    test_data = ImageFolderWithPaths(root=f"{data_dir}\\test_cleaned", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_data.classes