import torch
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data
import os
import math
import clip
import argparse

# 定义CIFAR - 100的类别名称
cifar100_classes = [
    'apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 
    'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 
    'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 
    'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 
    'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 
    'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 
    'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 
    'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 
    'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine',
    'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose',
    'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake',
    'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper',
    'table', 'tank', 'telephone', 'television', 'tiger', 'tractor',
    'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree',
    'wolf', 'woman', 'worm'
]

imagenet_single_template = [
    'a photo of a {}.',
]

imagenet_7_templates = [
    'itap of a {}.',
    'a origami {}.',
    'a bad photo of the {}.',
    'a photo of the large {}.',
    'a {} in a video game.',
    'art of the {}.',
    'a photo of the small {}.',
]


def zeroshot_classifier(clip, model, classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in classnames:
            texts = [template.format(classname) for template in templates]
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).cuda()
    return zeroshot_weights


def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def main():
    parser = argparse.ArgumentParser(description='CIFAR - 100 Zero - Shot Classification')
    parser.add_argument('--arch', type=str, default='ViT-B/16', help='CLIP model architecture')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers')
    parser.add_argument('--cw', type=float, default=0.5, help='Learning rate for weight update')
    parser.add_argument('--cr', type=float, default=0.20, help='Learning rate for rho update')
    parser.add_argument('--beta', type=float, default=0.8, help='Combination coefficient')
    parser.add_argument('--tau_t', type=float, default=0.01, help='Temperature for text logits')
    parser.add_argument('--tau_i', type=float, default=0.04, help='Temperature for image logits')
    parser.add_argument('--alpha', type=float, default=0.1, help='Regularization coefficient')
    parser.add_argument('--repeat', type=int, default=5, help='Number of repetitions for online zero - shot transfer')
    args = parser.parse_args()
    print(args)

    print('load pre - trained model')
    model, preprocess = clip.load(args.arch)
    model = model.cuda()
    model.eval()

    print('load data')
    # 加载CIFAR - 100数据集
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_set = datasets.CIFAR100(root='./data', train=False, download=True, transform=preprocess)
    loader = torch.utils.data.DataLoader(val_set, batch_size=args.batch_size, num_workers=args.workers)

    with torch.no_grad():
        image_feat = []
        image_label = []
        for i, (images, target) in enumerate(loader):
            images = images.cuda()
            target = target.cuda()
            image_features = model.encode_image(images)
            image_feat.append(F.normalize(image_features, dim=1))
            image_label.append(target)
    image_feat = torch.cat(image_feat, dim=0)
    image_label = torch.cat(image_label, dim=0)
    n = len(image_label)
    image_feat = image_feat.float()

    print('obtain text proxy')
    text_classifier = zeroshot_classifier(clip, model, cifar100_classes, imagenet_7_templates)
    text_classifier = text_classifier.float()
    logits_t = image_feat @ text_classifier
    acc1, acc5 = accuracy(logits_t, image_label, topk=(1, 5))
    top1 = (acc1 / n) * 100
    print(f'accuracy with text proxy: {top1:.2f}')

    print('online zero - shot transfer: repeat {} times'.format(args.repeat))
    num_class = len(torch.unique(image_label))
    acc_onzeta = torch.zeros(args.repeat).cuda()
    acc_onlab = torch.zeros(args.repeat).cuda()
    for iter in range(args.repeat):
        idx = torch.randperm(n).cuda()
        combo_label = torch.zeros(n, num_class).cuda()
        text_label = torch.zeros(n, num_class).cuda()
        w = text_classifier.clone()
        rho = torch.zeros(num_class).cuda()
        for i in range(n):
            lr = args.cw / math.sqrt(i + 1)
            rlr = args.cr / math.sqrt(i + 1)
            beta = args.beta * math.sqrt((i + 1) / n)
            x = image_feat[idx[i], :]
            tlabel = F.softmax(x @ text_classifier / args.tau_t, dim=0)
            tlabel = tlabel * torch.exp(rho)
            tlabel /= torch.sum(tlabel)
            rho -= rlr * (tlabel - args.alpha / num_class)
            rho[rho < 0] = 0
            text_label[i, :] = tlabel
            vision_label = F.softmax(x @ w / args.tau_i, dim=0)
            combo_label[i, :] = beta * vision_label + (1 - beta) * tlabel
            grad = torch.outer(x, vision_label - tlabel)
            w -= (lr / args.tau_i) * grad
            w = F.normalize(w, dim=0)
        acc1, acc5 = accuracy(text_label, image_label[idx], topk=(1, 5))
        acc_onlab[iter] = (acc1 / n) * 100
        acc1, acc5 = accuracy(combo_label, image_label[idx], topk=(1, 5))
        acc_onzeta[iter] = (acc1 / n) * 100
    print('mean acc of onlab is: {:.2f}'.format(torch.mean(acc_onlab)))
    print('mean acc of onzeta is: {:.2f}'.format(torch.mean(acc_onzeta)))


if __name__ == '__main__':
    main()
    
