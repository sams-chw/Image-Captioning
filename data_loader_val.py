import nltk
from nltk.translate import bleu_score
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

def evaluate_caption(ids_sample, params):
    bleu4_all = []
    print(params[data_loader])
    
    for key in tqdm(ids_sample, leave = False, position = 0):
        _, image, _, captions, _ = params[data_loader].dataset[key]

        reference = []
        for caption in captions:
            token = nltk.tokenize.word_tokenize(str(caption).lower())
            reference.append(token) 

        image = image.to(params[device])
        features = params[encoder](image.unsqueeze(0))
        output = params[decoder].sample(features.unsqueeze(0))

        sentence = clean_sentence(data_loader, output)
        hypothesis = nltk.tokenize.word_tokenize(str(sentence).lower())

        bleu4 = bleu_score.sentence_bleu(references=reference,
                                    hypothesis=hypothesis,
                                    smoothing_function=bleu_score.SmoothingFunction().method4)
        bleu4_all.append(bleu4)
        
    return bleu4_all
    

def clean_sentence(data_loader, output):
    sentence = ''
    for word in [data_loader.dataset.vocab.idx2word.get(x) for x in output]:
        if word == '<end>':
            break
        if word != '<start>' and word != '.' :
            sentence += ' ' + word
        if word == '.':
            sentence += word 
    sentence = sentence[1].upper() + sentence[2:]
    return sentence

def get_loader(transform,
               mode='train',
               batch_size=1,
               vocab_threshold=None,
               vocab_file='./vocab.pkl',
               start_word="<start>",
               end_word="<end>",
               unk_word="<unk>",
               vocab_from_file=True,
               num_workers=0,
               cocoapi_loc='/opt'):
    """Returns the data loader.
    Args:
      transform: Image transform.
      mode: One of 'train', 'val' or 'test'.
      batch_size: Batch size (if in validation or testing mode, must have batch_size=1).
      vocab_threshold: Minimum word count threshold.
      vocab_file: File containing the vocabulary. 
      start_word: Special word denoting sentence start.
      end_word: Special word denoting sentence end.
      unk_word: Special word denoting unknown words.
      vocab_from_file: If False, create vocab from scratch & override any existing vocab_file.
                       If True, load vocab from from existing vocab_file, if it exists.
      num_workers: Number of subprocesses to use for data loading 
      cocoapi_loc: The location of the folder containing the COCO API: https://github.com/cocodataset/cocoapi
    """
    
    assert mode in ['train', 'val', 'test'], "mode must be one of 'train', 'val' or 'test'."
    if vocab_from_file==False: assert mode=='train', "To generate vocab from captions file, must be in training mode (mode='train')."

    # Based on mode (train, val, test), obtain img_folder and annotations_file.
    if mode == 'train':
        if vocab_from_file==True: assert os.path.exists(vocab_file), "vocab_file does not exist.  Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json')
        
    if mode == 'val':
        assert batch_size==1, "Please change batch_size to 1 if validating your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/val2014/')
        instances_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/instances_val2014.json')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_val2014.json')
        
    if mode == 'test':
        assert batch_size==1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        assert vocab_from_file==True, "Change vocab_from_file to True."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')

    # COCO caption dataset.
    dataset = CoCoDataset(transform=transform,
                          mode=mode,
                          batch_size=batch_size,
                          vocab_threshold=vocab_threshold,
                          vocab_file=vocab_file,
                          start_word=start_word,
                          end_word=end_word,
                          unk_word=unk_word,
                          instances_file=instances_file,
                          annotations_file=annotations_file,
                          vocab_from_file=vocab_from_file,
                          img_folder=img_folder)

    if mode == 'train':
        # Randomly sample a caption length, and sample indices with that length.
        indices = dataset.get_train_indices()
        # Create and assign a batch sampler to retrieve a batch with the sampled indices.
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        batch_sampler = data.sampler.BatchSampler(sampler=initial_sampler,                                                                     batch_size=dataset.batch_size,                                                               drop_last=False)
        # data loader for COCO dataset.
        data_loader = data.DataLoader(dataset=dataset, 
                                      batch_sampler=batch_sampler,
                                      num_workers=num_workers)
                                    
    else:
        data_loader = data.DataLoader(dataset=dataset,
                                      batch_size=dataset.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)

    return data_loader

class CoCoDataset(data.Dataset):
   
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, 
        end_word, unk_word, instances_file, annotations_file, vocab_from_file, img_folder):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
            end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder
        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())
            print('Obtaining caption lengths...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[self.ids[index]]['caption']).lower()) for index in tqdm(np.arange(len(self.ids)))]
            self.caption_lengths = [len(token) for token in all_tokens]
        elif self.mode == 'val':
            self.coco = COCO(instances_file)
            self.ids = list(self.coco.anns.keys())
            self.coco_caps = COCO(annotations_file)
            self.ids_caps = list(self.coco_caps.anns.keys())
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            #print(image.shape)
            image = self.transform(image)
            #print(image.shape)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        elif self.mode == 'val':
            img_id = self.coco.anns[self.ids[index]]['image_id']
            #print("image id:", img_id)
            file_name = self.coco.loadImgs(img_id)[0]['file_name']
            #print("image file name:", file_name)
            
            PIL_image = Image.open(os.path.join(self.img_folder, file_name)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            annIds = self.coco_caps.getAnnIds(imgIds=img_id);
            #print("caption id:", annIds)

            caption = []
            for index in range(len(self.ids_caps)):
                if self.coco_caps.anns[self.ids_caps[index]]['image_id'] == img_id:
                    caption.append(self.coco_caps.anns[self.ids_caps[index]]['caption'])
               
            return (orig_image, image, img_id, caption, file_name)
            
            
        # obtain image if in test mode
        else:
            path = self.paths[index]
            print(self.img_folder, path)

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
       # print(sel_length)
        all_indices = np.where([self.caption_lengths[i] == sel_length for i in np.arange(len(self.caption_lengths))])[0]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train' or 'val':
            return len(self.ids)
        elif self.mode == 'test':
            return len(self.paths)