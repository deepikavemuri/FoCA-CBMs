import matplotlib.pyplot as plt
import torch.optim as optim
import os
import torch
from tqdm import tqdm
import math
import warnings
import clip
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import csv
import shutil
from matplotlib import rc
from data_utils import get_loaders, get_concepts, get_feature_dir

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica'], 'size': 22})
rc('text', usetex=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_optimizer(args, cparameters, dparameters):
    """
    Set the optimizer for the model; currently supports only Adam.
    :param args: the current setup arguments to get the learning rate and decay.
    :param parameters: the parameters to optimize; either a list of sets of parameters or a single set.

    :return: the optimizer object for training.
    """

    if isinstance(cparameters, list):
        new_params = []
        for i, cparams in enumerate(cparameters):
            new_params.append(
                {
                    'params': cparams,
                    'lr': args.learning_rate * (1. if i == 0 else 10.)
                 }
            )

        for i, dparams in enumerate(dparameters):
                new_params.append(
                {
                    'params': dparams,
                    'lr': args.learning_rate * args.multiplier_lr * (1. if i==0 else 10.),
                    'weight_decay': 0.
                })
    else:
        raise ValueError('Parameters of the classifier should be a list..'
                         )
    optimizer = optim.AdamW(new_params, lr=args.learning_rate,
                            weight_decay=args.weight_decay)

    return optimizer


def get_save_names(clip_name, d_probe, concept_set, save_dir):
    clip_save_name = "{}/batches/{}_{}.pt".format(save_dir, d_probe, clip_name.replace('/', ''))
    concept_set_name = (concept_set.split("/")[-1]).split(".")[0]
    text_save_name = "{}/{}_{}.pt".format(save_dir, concept_set_name, clip_name.replace('/', ''))

    return clip_save_name, text_save_name


def compute_and_save_features(args, model, preprocess,
                              device="cuda", only_text=False,
                              patchify=False, patch_size=(3, 3)):
    """

    :param val:
    :param patchify:
    :param only_text:
    :param args:
    :param model:
    :param dataset:
    :param text:
    :param base_dir:
    :param preprocess:
    :param save_frequency:
    :param device:
    :param patch_size:
    :return:
    """
    d1, d2, _ = get_loaders(args, patchify=patchify, batch_size=args.batch_size)
    datasets = [d1, d2]
    text = get_concepts(args.concept_name, patchify=patchify)
    text = torch.cat([clip.tokenize(f"{c}") for c in text]).to(device)

    for i, dataset in enumerate(datasets):

        val = False if i == 0 else True
        save_dir = get_feature_dir(args, val=val)
        os.makedirs(save_dir, exist_ok=True)
        save_name_features = save_dir + 'image_{}_{}'.format('feats',
                                                             'patches_{}_{}'.format(patch_size[0], patch_size[1])
                                                             if patchify else 'whole')
        save_name_texts = save_dir + '{}_{}_level_text_features.pt'.format(args.concept_name,
                                                                           'high' if not patchify else 'low')

        if os.path.exists(save_name_features):
            warnings.warn('File for features already exists.')
            return

        if not val:
            text_features = get_clip_text_features(model, text).float()
            torch.save(text_features, save_name_texts)

        if only_text:
            return

        with torch.no_grad():
            batch_num = 1
            print(len(dataset))
            for images, labels in tqdm(dataset):
                # print(device)
                # print(images.shape)
                if patchify:
                    # do the same for each patch
                    # if the original is something like [1,3,224,224] and take patches /2, it will be [1,3,112,112]
                    bs, channels, height, width = images.shape
                    patch_height, patch_width = int(height / patch_size[0]), int(width / patch_size[1])
                    stride_height, stride_width = patch_height, patch_width
                    patches = patchify_custom(images, (patch_height, patch_width), (stride_height, stride_width)).to(
                        device)

                    or_shape = patches.shape

                    # visualize_patches(patches[0].cpu())
                    # we should just reshape batch size to be images * patches and then do regular stuff
                    new_shape = [-1, or_shape[3], or_shape[4], or_shape[5]]
                    feats = patches.reshape(new_shape).permute(0, 3, 1, 2)
                    # print(feats.shape)
                else:
                    feats = images

                feats = preprocess(feats)
                features = model.encode_image(feats.to(device)).float()

                if batch_num == 1:
                    feats_enc = features.cpu()
                    labels_batch = labels.cpu()
                else:
                    feats_enc = torch.cat([feats_enc, features.cpu()], 0)
                    labels_batch = torch.cat([labels_batch, labels.cpu()], 0)

                batch_num += 1

        # we have some data left on the list
        torch.save((feats_enc if not patchify else
                    feats_enc.reshape(-1, patch_size[0], patch_size[1], feats_enc.shape[-1]),
                    labels_batch),
                   save_name_features + '.pt')
        # free memory
        del feats_enc
        del labels_batch
        torch.cuda.empty_cache()

        if patchify:
            with open(save_dir + 'patches_sizes_{}_{}.txt'.format(patch_size[0], patch_size[1]), 'w') as f:
                f.write(str(or_shape[1]) + ' ' + str(or_shape[2]))

    return


def get_clip_text_features(model, text, batch_size=1000):
    """
    gets text features without saving, useful with dynamic concept sets
    """
    text_features = []
    with torch.no_grad():
        for i in range(math.ceil(len(text) / batch_size)):
            text_features.append(model.encode_text(text[batch_size * i:batch_size * (i + 1)]))
    text_features = torch.cat(text_features, dim=0)
    return text_features


def _make_save_dir(save_name):
    """
    creates save directory if one does not exist
    save_name: full save path
    """
    save_dir = save_name[:save_name.rfind("/")]
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    return


def patchify_custom(image, patch_size, stride):
    """
    Patchify an image with given patch size and stride.

    :param image: Tensor. The image to patchify. Assumes a 4-dimensional input.
    :param patch_size: Tuple. The height and width of each patch.
    :param stride: Tuple. The height and width of the stride.

    :return: Tensor. The patches of the original image given the patch size and stride parameters.
    """

    patch_height, patch_width = patch_size
    stride_height, stride_width = stride

    patches = image.unfold(2, patch_height, stride_height).unfold(3, patch_width, stride_width).permute(0, 2, 3, 4, 5,
                                                                                                        1)

    return patches


def visualize_patches(patches, titles=['random'], path = None):
    '''

    :param patches: the tensor with the patches. It must have the following form:
                num_patches x num_patches x 1 x patch_size x patch_size x channels
    :return: Nothing, just plot
    '''
    print('In the viz function', patches.shape)
    # num_patches, patch_chann, patch_height, patch_width = patches.shape
    # channels, or_height, or_width = orig_image.shape
    num_rows, num_cols = patches.shape[0], patches.shape[1]
    print(num_rows, num_cols)
    if len(titles) == 1:
        titles = titles * (num_rows * num_cols)

    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12, 12))
    count = 0
    for i in range(num_rows):
        for j in range(num_cols):
            cur_patch = patches[i, j]
            ax = axs[i, j]
            print(int(i / num_cols), int(i % num_cols))
            ax.imshow(cur_patch, interpolation='none')
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth(1)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.set_title(titles[count])
            count += 1
    plt.grid()
    plt.subplots_adjust(wspace=0.01, hspace=0.01,

                        )
    fig.tight_layout()


    #plt.show()
    if path is not None:
        plt.axis('off')
        plt.savefig(path, bbox_inches='tight')


def bin_accuracy(output, target, threshold = 0.5):
    correct = (target.eq(output.gt(threshold)).float()).mean()
    return [correct*(100.0)]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def bin_concrete_sample(a, temperature, eps=1e-8):
    """"
    Sample from the binary concrete distribution
    """

    U = torch.rand_like(a).clamp(eps, 1. - eps)
    L = torch.log(U) - torch.log(1. - U)
    X = torch.sigmoid((L + a) / temperature)

    return X

def concrete_sample(a, temperature,  eps=1e-8, axis=-1):
    """
    Sample from the concrete relaxation.
    :param a: torch tensor: logits of the concrete relaxation
    :param temperature: float: the temperature of the relaxation
    :param eps: float: eps to stabilize the computations
    :param axis: int: axis to perform the softmax of the gumbel-softmax trick
    :return: a sample from the concrete relaxation with given parameters
    """

    def _gen_gumbels():
        U = torch.rand(a.shape, device=a.device)
        gumbels = - torch.log(- torch.log(U + eps) + eps)
        if torch.isnan(gumbels).sum() or torch.isinf(gumbels).sum():
            # to avoid zero in exp output
            gumbels = _gen_gumbels()
        return gumbels

    gumbels = _gen_gumbels()  # ~Gumbel(0,1)
    a = (a  + gumbels)/temperature

    y_soft = a.softmax(axis)

    index = y_soft.max(axis, keepdim=True)[1]
    y_hard = torch.zeros_like(y_soft).scatter_(axis, index, 1.0)
    ret = (y_hard - y_soft).detach() + y_soft

    if torch.isnan(ret).sum():
        raise OverflowError(f'gumbel softmax output: {ret}')

    return ret


def concept_activation_per_class(loader, disc, num_concepts,
                                 num_classes, concepts, classes,
                                 base_dir='saved_models', prior=True):
    conc_act = torch.zeros([num_concepts, num_classes])
    disc.eval()
    with torch.no_grad():
        count_per_class = np.zeros([num_classes])
        for batch, (images, labels) in enumerate(loader):
            labels = labels.type(torch.LongTensor)  # casting to long

            images = images.to('cuda', non_blocking=True)
            labels = labels

            feats = images  # / images.norm(dim=-1, keepdim=True)

            # probs in R^{batch, concepts}
            probs, _ = disc(feats, probs_only=True)
            probs = probs.cpu().numpy()
            # set the very low probabilities to zero
            probs[np.where(probs < 1e-2)] = 0.

            if np.any(probs > 1.):
                raise ValueError('what, probs >1?')

            # add the contribution of each example to the respective entries

            for i in range(num_classes):
                batch_class_i = probs[np.where(labels == i)]
                count_per_class[i] += batch_class_i.shape[0]
                conc_act[:, i] += batch_class_i.sum(0)
        conc_act /= count_per_class

        results_dir = base_dir + 'prior_figs/' * prior + 'posterior_figs/' * (not prior)
        os.makedirs(results_dir, exist_ok=True)

        np.savetxt(results_dir + "concept_acts_per_class.csv", conc_act,
                   delimiter=",", fmt='%10.5f')
        # concept per fig
        cpf = 40
        for i in range(int(conc_act.shape[0] / cpf) + 1):
            cur_act = conc_act[i * cpf: (i + 1) * cpf]
            cur_conc = concepts[i * cpf: (i + 1) * cpf]
            lencur = cur_act.shape[0]

            fig = plt.figure(figsize=(15, 15))
            ax = sns.heatmap(cur_act.T, cmap='binary', linewidth=0.5,
                             square=True, vmin=0.0, vmax=1.0,
                             cbar_kws={"shrink": .95, "ticks": [0.0, 0.25, 0.5, 0.75, 1.0],
                                       'aspect': 50, 'pad': 0.01})

            plt.xticks(np.arange(lencur) + 0.5, cur_conc,
                       rotation=90, fontsize="8")
            plt.yticks(np.arange(num_classes) + 0.5, classes,
                       rotation=0, fontsize="8", va="center")

            # ax.set_yticklabels(classes, minor=True)
            plt.tight_layout()
            plt.savefig(results_dir + 'batch_{}.pdf'.format(i), bbox_inches="tight")
            plt.close(fig)


def concept_activation_per_example(loader, disc, classifier, num_concepts,
                                   concepts, classes, text_features, dataset,
                                   base_dir='saved_models', ):
    disc.eval()
    classifier.eval()

    # inds = np.random.choice(len(loader.dataset), 10)

    data = loader.dataset
    inds = np.random.choice(np.arange(13950, 13953, 1), 3, replace=False)
    results_dir = base_dir + 'post_analysis/'
    os.makedirs(results_dir, exist_ok=True)

    indices = 'imagenet_indices.csv' if 'imagenet' in dataset else 'cub_indices.csv'
    with open(indices, 'r') as f:
        csvreader = csv.reader(f)
        count = 0
        fpaths = {}
        for row in csvreader:
            if count == 0:
                count = 1
                continue
            fpaths[int(row[0])] = row[1]

    # for each example get the decision, get the weights corresponding to the class
    # and then get the concept contribution from that
    with torch.no_grad():
        text_features /= text_features.norm(dim=-1, keepdim=True)
        for ind in inds:

            cur_feats, cur_label = data[ind]
            print(ind)
            print(cur_feats.shape)
            print(cur_label)
            cur_feats = cur_feats.to('cuda')
            cur_label = cur_label.to('cuda')
            mask, _ = disc(cur_feats, probs_only=True)
            cur_feats /= cur_feats.norm(dim=-1, keepdim=True)
            similarity = (cur_feats @ text_features.T)

            pred = torch.nn.functional.softmax(classifier(similarity, mask=mask), dim=-1)

            class_ind = torch.argmax(pred)
            class_name = classes[class_ind]
            class_name = ''.join([i for i in class_name if (not i.isdigit()) and (i != '.')])
            class_name = class_name.replace('_', ' ')
            true_class = classes[int(cur_label.cpu().numpy())]
            true_class = ''.join([i for i in true_class if not i.isdigit() and (i != '.')])
            true_class = true_class.replace('_', ' ')

            if class_name != true_class:
                continue

            spec_dir = results_dir + 'ind_{}/'.format(ind)
            os.makedirs(spec_dir, exist_ok=True)

            # get the original image if cub or imagenet
            if dataset in ['imagenet', 'cub']:
                fpath = fpaths[ind]
                fname = fpath.split('/')[-1]

                shutil.copyfile(fpath, spec_dir + fname)

            mask[torch.where(mask < 0.01)] = 0.

            # get the relevant weights and use the mask to filter
            # the mask should be probs to reflect the contribution
            rel_weights = (classifier.W[:, class_ind] * mask).cpu().numpy()
            rel_bias = classifier.bias[class_ind].cpu().numpy()
            rel_conf = pred[class_ind].cpu().numpy()
            mask = mask.cpu().numpy().astype(np.float32)
            similarity = similarity.cpu().numpy()
            contribution = similarity * rel_weights
            sparsity = (mask > 0.01).sum() / num_concepts

            sort_inds = np.argsort(contribution)[::-1]
            print(len(sort_inds))
            print(num_concepts)
            print(len(mask))
            print(len(similarity))
            print(len(rel_weights))
            print(len(contribution))
            print(len(concepts))

            with open(spec_dir + 'stats.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])
                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                     rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', (mask > 0.).sum() / num_concepts])

            with open(spec_dir + 'stats_only_active.csv', 'w') as f:
                writer = csv.writer(f)
                writer.writerow(['True Class', 'Class Prediction', 'Conf', 'Class Bias'])
                writer.writerow([true_class, class_name, rel_conf, rel_bias])

                writer.writerow(['Concept', 'Active', 'Similarity', 'Weight', 'Contr (S*W)'])
                for i in range(num_concepts):
                    if mask[sort_inds[i]] > 0.01:
                        writer.writerow([concepts[sort_inds[i]], mask[sort_inds[i]], similarity[sort_inds[i]],
                                         rel_weights[sort_inds[i]], contribution[sort_inds[i]]])
                writer.writerow(['Sparsity', sparsity])

            plt.rc('text', usetex=True)
            plt.rc('font', family='serif')
            fig, ax = plt.subplots(layout='constrained', figsize=(18, 6))

            # Example data
            concepts_top = (concepts[sort_inds[0]],
                            concepts[sort_inds[1]],
                            concepts[sort_inds[2]],
                            concepts[sort_inds[3]],
                            'NOT ' * (contribution[sort_inds[-4]] < 0) + concepts[sort_inds[-4]],
                            'NOT ' * (contribution[sort_inds[-3]] < 0) + concepts[sort_inds[-3]],
                            'NOT ' * (contribution[sort_inds[-2]] < 0) + concepts[sort_inds[-2]],
                            'NOT ' * (contribution[sort_inds[-1]] < 0) + concepts[sort_inds[-1]]
                            )
            y_pos = np.arange(len(concepts_top))
            contribution = np.abs(contribution)
            contrib = contribution[sort_inds[:4]].tolist() + contribution[sort_inds[-4:]].tolist()
            colors = ['darkkhaki'] * 4 + ['darksalmon'] * 4
            hbars = ax.barh(y_pos, contrib, align='center',
                            height=0.5, color=colors)
            ax.set_yticks(y_pos, labels=concepts_top)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Concept Contribution')
            ax.set_title('Pred: {} - Conf: {:.5f} - Sparsity: {:.2f}\%'.format(class_name, rel_conf, 100 * sparsity))
            ax.bar_label(hbars, fmt='%.4f')
            ax.margins(x=0.1)

            plt.tight_layout()
            plt.savefig(spec_dir + 'contribution.pdf')  # , bbox_inches="tight")
            plt.close(fig)


if __name__ == '__main__':
    data_train, data_val, classes = data_loader('places365')
    print(data_train)
    print(classes)
