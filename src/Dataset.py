import numpy as np
from PIL import Image
from random import sample, random
from torch.utils.data.sampler import SubsetRandomSampler
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, Dataset

def random_crops(img, k):
    crops=[]
    five256=torchvision.transforms.RandomCrop(256)
    for j in range(k):
        im = five256(img)
        crops.append(im)
    return crops

def crops_and_random( img):
    im=np.array(img)
    w, h, _ = im.shape
    if w<h:
        wi=512
        hi=int(wi*h*1.0/w)
    else:
        hi=512
        wi = int(hi * w * 1.0 / h)

    res=torchvision.transforms.Resize((wi,hi),interpolation=2)
    img=res(img)
    Rand=torchvision.transforms.RandomCrop(512)
    re=torchvision.transforms.Resize((256,256),interpolation=2)
    return random_crops(img, 4)+[re(Rand(img))]

def val_crops(img):
    im = np.array(img)
    w, h, _ = im.shape
    if w < h:
        wi = 512
        hi = int(wi * h * 1.0 / w)
    else:
        hi = 512
        wi = int(hi * w * 1.0 / h)
    res=torchvision.transforms.Resize((wi,hi),interpolation=2)
    img=res(img)
    im=np.array(img)
    Rand = torchvision.transforms.RandomCrop(512)
    re = torchvision.transforms.Resize((256, 256), interpolation=2)
    a=int(wi/256)
    b=int(hi/256)
    crs=[]
    for i in range(a):
        for j in range(b):
            crs.append(Image.fromarray((im[i*256:((i+1)*256),j*256:((j+1)*256)]).astype('uint8')).convert('RGB'))
    return sample(crs,4)+[re(Rand(img))]

def RandomErase( img, p=0.5, s=(0.06,0.12), r=(0.5,1.5)):
    im=np.array(img)
    w,h,_=im.shape
    S=w*h
    pi=random()
    if pi>p:
        return img
    else:
        Se=S*(random()*(s[1]-s[0])+s[0])
        re=random()*(r[1]-r[0])+r[0]
        He=int(np.sqrt(Se*re))
        We=int(np.sqrt(Se/re))
        if He>=h:
            He=h-1
        if We>=w:
            We=w-1
        xe=int(random()*(w-We))
        ye=int(random()*(h-He))
        im[xe:xe+We,ye:ye+He]=int(random()*255)
        return Image.fromarray(im.astype('uint8')).convert('RGB')

def default_loader(path):
    return Image.open(path).convert('RGB')

def load_taxonomy(ann_data, tax_levels, classes):
    # loads the taxonomy data and converts to ints
    taxonomy = {}

    if 'categories' in ann_data.keys():
        num_classes = len(ann_data['categories'])
        for tt in tax_levels:
            tax_data = [aa[tt] for aa in ann_data['categories']]
            _, tax_id = np.unique(tax_data, return_inverse=True)
            taxonomy[tt] = dict(zip(range(num_classes), list(tax_id)))
    else:
        # set up dummy data
        for tt in tax_levels:
            taxonomy[tt] = dict(zip([0], [0]))

    # create a dictionary of lists containing taxonomic labels
    classes_taxonomic = {}
    for cc in np.unique(classes):
        tax_ids = [0]*len(tax_levels)
        for ii, tt in enumerate(tax_levels):
            tax_ids[ii] = taxonomy[tt][cc]
        classes_taxonomic[cc] = tax_ids

    print(classes_taxonomic)
    print(taxonomy ,'\n\n\n\n\n\n')

    return taxonomy, classes_taxonomic

class INAT(torch.utils.data.Dataset):
    def __init__(self, root, ann_file, is_train=True):

        # load annotations
        print('Loading annotations from: ' + os.path.basename(ann_file))
        with open(ann_file) as data_file:
            ann_data = json.load(data_file)

        # set up the filenames and annotations
        self.imgs = [aa['file_name'] for aa in ann_data['images']]
        self.ids = [aa['id'] for aa in ann_data['images']]

        # if we dont have class labels set them to '0'
        if 'annotations' in ann_data.keys():
            self.classes = [aa['category_id'] for aa in ann_data['annotations']]        
        else:
            self.classes = [0]*len(self.imgs)

        # load taxonomy
        self.tax_levels = ['id', 'genus', 'family', 'order', 'class', 'phylum', 'kingdom']
                           #8142, 4412,    1120,     273,     57,      25,       6
        self.taxonomy, self.classes_taxonomic = load_taxonomy(ann_data, self.tax_levels, self.classes)

        # print out some stats
        print( '\t' + str(len(self.imgs)) + ' images' )
        print( '\t' + str(len(set(self.classes))) + ' classes' )

        self.root = root
        self.is_train = is_train
        self.loader = default_loader

        # # augmentation params
        self.im_size = [256, 256]  # can change this to train on higher res
        self.mu_data = [0.485, 0.456, 0.406]
        self.std_data = [0.229, 0.224, 0.225]
        self.brightness = 0.4
        self.contrast = 0.4
        self.saturation = 0.4
        self.hue = 0.25

        # augmentations
        self.center_crop = transforms.CenterCrop((self.im_size[0], self.im_size[1]))
        self.scale_aug = transforms.RandomResizedCrop(size=self.im_size[0])
        self.flip_aug = transforms.RandomHorizontalFlip()
        self.color_aug = transforms.ColorJitter(self.brightness, self.contrast, self.saturation, self.hue)
        self.tensor_aug = transforms.ToTensor()
        self.norm_aug = transforms.Normalize(mean=self.mu_data, std=self.std_data)

        self.train_compose = transforms.Compose([
            transforms.Lambda(lambda img: RandomErase(img)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.Lambda(lambda img: crops_and_random(img)),
            #transforms.Resize((512,512),interpolation=2),
            #transforms.Lambda(lambda img: four_and_random(img)),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])])(crop) for
                                                        crop in crops]))

            ])

        self.valid_compose = transforms.Compose([
            transforms.Lambda(lambda img: val_crops(img)),
            transforms.Lambda(lambda crops: torch.stack([transforms.Compose([transforms.ToTensor(),
                                                                            transforms.Normalize(mean=[n / 255.
                                                                                                        for n in
                                                                                                        [129.3, 124.1,
                                                                                                        112.4]],
                                                                                                std=[n / 255. for n in
                                                                                                    [68.2, 65.4,
                                                                                                        70.4]])])(crop) for
                                                        crop in crops]))

            ])

    def __getitem__(self, index):
        path = self.root + self.imgs[index]
        im_id = self.ids[index]
        img = self.loader(path)
        species_id = self.classes[index]
        tax_ids = self.classes_taxonomic[species_id]

        if self.is_train:
            # img = self.scale_aug(img)
            # img = self.flip_aug(img)
            # img = self.color_aug(img)
            img = self.train_compose(img)

        else:
            img = self.valid_compose(img)


        # print("end of __getitem__")
        # print("type(img)", type(img))
        # print("img", img)
        # img = self.tensor_aug(img)
        # img = self.norm_aug(img)

        # return img, im_id, species_id, tax_ids
        # img = torch.tensor(img)
        return img, species_id


    def __len__(self):
        return len(self.imgs)

# RGB one set