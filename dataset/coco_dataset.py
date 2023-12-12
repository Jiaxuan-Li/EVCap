import os
from PIL import Image
import json
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset


class COCODataset(Dataset):

    def __len__(self) -> int:
        return len(self.annotation)

    def __getitem__(self, index):
        ann = self.annotation[index]
        img_file = f'COCO_train2014_{int(ann["image_id"]):012d}.jpg'
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption = ann["caption"]
        return {
            "image": image,
            "text_input": caption,
            "image_id": self.img_ids[ann["image_id"]],
        }

    def __init__(self, data_root):
        ann_path = os.path.join(data_root, 'annotations/captions_train2014.json')
        self.vis_root=os.path.join(data_root, 'train2014')
        self.annotation = []
        self.annotation.extend(json.load(open(ann_path, "r"))['annotations'])
        self.img_ids = {}
        n = 0
        for ann in self.annotation:
            img_id = ann["image_id"]
            if img_id not in self.img_ids.keys():
                self.img_ids[img_id] = n
                n += 1
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
