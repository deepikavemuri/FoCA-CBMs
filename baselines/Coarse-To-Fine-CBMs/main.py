import os
import sys
import time
import clip
import torch
import torch.nn as nn

import utils
from clip.clip import _transform
import warnings
import argparse
from networks import LinearClassifier, DiscoveryMechanism
import tensorboard_logger as tlogger
from utils import get_optimizer, compute_and_save_features, AverageMeter, accuracy
from data_utils import get_loaders, get_concepts, get_feature_dir, get_concept_indicators
import shutil
from torch.utils.tensorboard import SummaryWriter
import itertools

parser = argparse.ArgumentParser(description='Settings for the CF CBM')
#
parser.add_argument("--dataset", type=str, default="cifar10")
parser.add_argument("--clip_version", type=str, default="RN50")
parser.add_argument("--concept_name", type=str, default="cifar100")
parser.add_argument("--device", type=str, default="cuda",
                    help="Which device to use")
parser.add_argument("--save_dir", type=str, default='saved_models',
                    help="where to save the results")
parser.add_argument('--compute_similarities', action='store_true',
                    help='compute, save and use similarities for the given dataset')
parser.add_argument('--optimizer', default='adam', help='the considered optimizer')
parser.add_argument('--print_freq', type=int, default=100,
                    help='print frequency')
# training args
parser.add_argument('--epochs', type=int, default=400,
                    help='number of epochs')
parser.add_argument('--num_workers', type=int, default=8,
                    help='num of workers to use')
parser.add_argument('--learning_rate', type=float, default=3e-4,
                    help='learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-5,
                    help='weight decay value')
parser.add_argument("--batch_size", type=int, default=128,
                    help="The training batch size")
parser.add_argument('--save_freq', type=int, default=50,
                    help='save frequency')
parser.add_argument('--resume', type=str, default=None,
                    help='checkpoint to resume training')
parser.add_argument('--patch_size', type=list, default=[3, 3],
                    help='the split sizes for the image.')

parser.add_argument('--patchify', action='store_true', help='Whether or not to patchify the image.')
parser.add_argument('--reduction', default='max', help='The reduction operation for the low level')
parser.add_argument('--low_level_only', action='store_true', help='use only patches for classification')
parser.add_argument('--discovery', action='store_true', help='Use data driven masking')
parser.add_argument('--only_images', action='store_true', help='Use data driven masking')
parser.add_argument('--tie_indicators', action='store_true', help='Tie the the high and low levels')
parser.add_argument('--eval', action='store_true', help='Load and eval the models only')


def set_clip_model(args, device='cuda'):
    # Load the model
    clip_model, _ = clip.load(args.clip_version, device)
    # print(clip_model)
    # exit(0)
    patch_transform = _transform(clip_model.visual.input_resolution, pil_flag=False, norm_flag=True)

    return clip_model, patch_transform


def train(args, loader, concept_sets, classifiers, criterion, optimizer, epoch, discoverers, binary_inds):
    btime, dtime, loss_low, loss_high, top1_low, top1_high = [AverageMeter() for _ in range(6)]
    start_time = time.time()
    nested_mask = 1.

    for batch, data in enumerate(loader):
        dtime.update(time.time() - start_time)
        total_loss = 0.  # torch.tensor([0.0], requires_grad=True).to(args.device)
        if len(data) == 2:
            data = [data]
        else:
            data = [data[:2], data[2:]]

        for level in range(len(data)):
            # if it's only one level we only need the following
            images = data[level][0].cuda()
            labels = data[level][1].type(torch.LongTensor)  # casting to long
            labels = labels.cuda()


            # texts and features are not normalized
            # this is for loading the data split into patches
            with torch.no_grad():
                text = concept_sets[level]

                text = text / text.norm(dim=-1, keepdim=True)
                feats = images / images.norm(dim=-1, keepdim=True)
                similarity = (feats @ text.T) if not args.only_images else images

            # this is the main logic, classifier is num_concepts x classes
            kl = 0.
            # import pdb; pdb.set_trace()
            if args.discovery:
                mask, kl = discoverers[level](images)
                # print("mask shape: ", mask.shape, binary_inds.shape)
                # this ties the levels together
                if level == 1 and args.tie_indicators:
                    # print("!!! ", nested_mask.shape, binary_inds.shape)
                    # nested_mask_proj = nested_mask @ binary_inds.T.to(nested_mask.device)
                    # nested_mask_proj = nested_mask_proj / nested_mask_proj.max()
                    # print(mask.shape, nested_mask_proj.shape)
                    mask *= nested_mask

                # binary inds is concepts_high x concepts_low
                # mask low level is batch_size x concepts_high
                elif (level == 0 and args.patchify) and not args.low_level_only:
                    # print("mask shape: ", mask.shape, binary_inds.shape)
                    nested_mask = (mask @ binary_inds) / binary_inds.shape[0]                 # changed to transpose

                    # reduce over the high level dimension, this should now be batch_size x num low level concs
                    nested_mask = nested_mask.view([nested_mask.shape[0],
                                                    1,
                                                    1,
                                                    nested_mask.shape[-1]])

                output = classifiers[level](similarity, mask=mask)

            else:
                output = classifiers[level](similarity)

            # that means that patchify is on
            if level == 1 or args.low_level_only:
                # we need a reduction; mean or max
                output = args.reduction_func(output, dim=[1, 2])

            # minimize the cross entropy loss
            loss = criterion(output, labels) + args.kl_scale * kl

            loss_high.update(loss.item(), output.shape[0]) if level == 0 and not args.low_level_only else \
                loss_low.update(loss.item(), output.shape[0])
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1_high.update(acc1[0], output.shape[0]) if level == 0 and not args.low_level_only else \
                top1_low.update(acc1[0], output.shape[0])
            total_loss += loss

        # optimization
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        btime.update(time.time() - start_time)
        start_time = time.time()

        # print info
        if (batch + 1) % args.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'lossHigh {losshigh.val:.3f} ({losshigh.avg:.3f})\t'
                  'lossLow {losslow.val:.3f} ({losslow.avg:.3f})\t'
                  'Acc@1High {top1high.val:.3f} ({top1high.avg:.3f})\t'
                  'Acc@1Low {top1low.val:.3f} ({top1low.avg:.3f})'.format(
                epoch, batch + 1, len(loader), batch_time=btime,
                data_time=dtime, losshigh=loss_high, losslow=loss_low,
                top1high=top1_high, top1low=top1_low))
            sys.stdout.flush()

    print(' Training Acc@1 High {top1high.avg:.3f} \t'
          'Training Acc@1 Low {top1low.avg:.3f}\n'.format(top1high=top1_high, top1low=top1_low))

    return loss_low.avg + loss_high.avg, top1_low.avg, top1_high.avg


def validate(args, loader, concept_sets, classifiers, criterion, discoverers, num_samples, binary_inds):
    btime, loss_low, loss_high, top1_low, top1_high = [AverageMeter() for _ in range(5)]
    avg_perc_active = [0. for _ in classifiers]
    nested_mask = 1.

    with torch.no_grad():
        end = time.time()
        for batch, data in enumerate(loader):
            if len(data) == 2:
                data = [data]
            else:
                data = [data[:2], data[2:]]

            for level in range(len(data)):

                images = data[level][0].to('cuda', non_blocking=True)
                labels = data[level][1].type(torch.LongTensor).to('cuda', non_blocking=True)

                text = concept_sets[level]
                # images = multiplier2(images)
                text = text / text.norm(dim=-1, keepdim=True)
                feats = images / images.norm(dim=-1, keepdim=True)
                similarity = (feats @ text.T) if not args.only_images else images

                if args.discovery:
                    output = 0.
                    for _ in range(num_samples):
                        mask, _ = discoverers[level](images, probs_only=True)

                        # this ties the levels together
                        if level == 1 and args.tie_indicators:

                            # mask *= nested_mask
                            # Made changes here from og code.
                            # nested_mask_proj = nested_mask @ binary_inds.to(nested_mask.device)
                            # nested_mask_proj = nested_mask_proj / nested_mask_proj.max()
                            mask *= nested_mask

                        elif (level == 0 and args.patchify) and not args.low_level_only:

                            nested_mask = mask @ binary_inds / binary_inds.shape[0]
                            # reduce over the high level dimension, this should now be batch_size x num low level concs
                            nested_mask = nested_mask.view([nested_mask.shape[0],
                                                            1,
                                                            1,
                                                            nested_mask.shape[-1]])
                        output += classifiers[level](similarity, mask) / num_samples
                        perc_active = (mask > 0.01).cpu().numpy().mean()
                        avg_perc_active[level] += perc_active

                else:
                    output = classifiers[level](similarity)

                if level == 1 or args.low_level_only:
                    # we need a reduction; mean or max
                    output = args.reduction_func(output, dim=[1, 2])

                loss = criterion(output, labels)

                # update metrics
                loss_high.update(loss.item(), output.shape[0]) if level == 0 and not args.low_level_only else \
                    loss_low.update(loss.item(), output.shape[0])
                acc1, acc5 = accuracy(output, labels, topk=(1, 5))
                top1_high.update(acc1[0], output.shape[0]) if level == 0 and not args.low_level_only else \
                    top1_low.update(acc1[0], output.shape[0])

                cur_perc_high = 1. if args.low_level_only or not args.discovery else avg_perc_active[0] / (batch + 1)
                cur_perc_low = 1. if not args.patchify or not args.discovery else avg_perc_active[-1] / (batch + 1)

            # measure elapsed time
            btime.update(time.time() - end)
            end = time.time()

            if batch % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'lossHigh {losshigh.val:.3f} ({losshigh.avg:.3f})\t'
                      'lossLow {losslow.val:.3f} ({losslow.avg:.3f})\t'
                      'Acc@1High {top1high.val:.3f} ({top1high.avg:.3f})\t'
                      'Acc@1Low {top1low.val:.3f} ({top1low.avg:.3f})\t'
                      'Avg Perc Act High {perc_high:.4f}\t'
                      'Avg Perc Act Low {perc_low:.4f}'.format(
                    batch, len(loader), batch_time=btime,
                    losshigh=loss_high, losslow=loss_low,
                    top1high=top1_high, top1low=top1_low,
                    perc_high=cur_perc_high, perc_low=cur_perc_low))

        avg_perc_active_high = 1. if args.low_level_only or not args.discovery else avg_perc_active[0] / (batch + 1.)
        avg_perc_active_low = 1. if not args.patchify or not args.discovery else avg_perc_active[-1] / (batch + 1.)

        print(' Validation Acc@1 High {top1high.avg:.3f} \t'
              'Validation Acc@1 Low {top1low.avg:.3f}\n\n'.format(top1high=top1_high, top1low=top1_low))

    return loss_low.avg + loss_high.avg, top1_low.avg, top1_high.avg, avg_perc_active_low, avg_perc_active_high


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        save_path_best = filename.split('.')[0] + '_best.pth.tar'
        shutil.copyfile(filename, save_path_best)


reduction_mapping = {
    'mean': torch.mean,
    'max': torch.amax,
    'sum': torch.sum
}
if __name__ == '__main__':
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() and args.device == 'cuda' else "cpu"

    if args.low_level_only and not args.patchify:
        raise ValueError('Cannot use only patches without turning on patchify..')

    concepts = [args.dataset]
    kl_scales = [1e-4]
    lrs = [0.005]
    priors = [1e-4, 1e-2, 1e-1, 1.]
    mults = [5.0, 10.]
    masks = [True]
    cversions = [args.clip_version]      #['ViT-B/16']
    reduct_funcs = ['max']
    low_only = [False]
    patchify_flags = [True]
    tied_indicators = [True]
    print(clip.available_models())


    # if true classify using only the clip image embeddings
    only_images = False

    params = [concepts, kl_scales, lrs, masks, priors, mults,
              cversions, reduct_funcs,
              low_only, patchify_flags, tied_indicators]
    for concept, kl_scale, lr, mask, prior, mult, \
            cversion, reduction, \
            low_level_only, patch_flag, tied_inds in itertools.product(*params):

        args.concept_name = concept
        args.discovery = mask
        args.learning_rate = lr
        args.prior = prior
        args.kl_scale = kl_scale
        args.multiplier_lr = mult
        args.only_images = only_images
        args.clip_version = cversion
        args.reduction_func = reduction_mapping[reduction]
        args.reduction = reduction
        args.patchify = patch_flag
        args.low_level_only = low_level_only
        args.tie_indicators = tied_inds
        if low_level_only or not patch_flag:
            args.tie_indicators = False
        if not patch_flag:
            if reduction == 'mean':
                continue

        if args.low_level_only and not args.patchify:
            continue

        if (args.low_level_only and tied_inds) or (not args.patchify and tied_inds):
            continue

        print(args)
        if only_images and mask:
            raise ValueError('Can\'t use only the image embeddings with discovery....')

        results_dir = utils.get_feature_dir(args) + "1"
        spec_dir = results_dir[:-6] + 'only_images/' * only_images + \
                   'low_level_only_' * args.low_level_only + \
                   'high_level_only_' * (not args.patchify) + \
                   '{}_{}/mask_{}_{}/{}_{}_multiplier_{}/'.format(
                       args.concept_name, args.reduction,
                       args.discovery, 'tied' if tied_inds else '', args.optimizer,
                       str(args.learning_rate).replace('.', ''),
                       str(args.multiplier_lr).replace('.', '_')
                   )
        if args.discovery:
            spec_dir += 'kl_scale_{}_prior_{}/'.format(str(args.kl_scale).replace('.', ''),
                                                       str(args.prior).replace('.', ''))


        if os.path.exists(spec_dir) and not args.eval:
            continue

        # moved this here. we can compute the similarities and continue training
        clip_model, preprocess = None, None
        if args.compute_similarities:
            clip_model, preprocess = set_clip_model(args)
            with torch.no_grad():

                # first compute the non-patchified embeddings
                if not args.low_level_only:
                    compute_and_save_features(args, clip_model, preprocess, device=device, patchify=False)

                if args.patchify:
                    compute_and_save_features(args, clip_model, preprocess, device=device, patchify=True)
            args.compute_similarities = False
        concepts = []
        num_concepts = []

        # need to load both the "regular" and the patch dataset
        # I changed this to reduce the complexity
        # check if something is wrong
        dual_data = not args.low_level_only and args.patchify
        args.dual_data = dual_data
        if dual_data:
            train_loader, val_loader, classes = get_loaders(args, batch_size=args.batch_size, dual_data=dual_data)
            concepts = [get_concepts(args.concept_name, False), get_concepts(args.concept_name, True)]
            binary_inds = get_concept_indicators(args.concept_name).to(device)
            num_concepts = [len(concepts[0]), len(concepts[1])]
            num_classes = len(classes)
            # print(concepts[0], concepts[1], binary_inds)
            # print(num_classes, num_concepts)

        else:
            train_loader, val_loader, classes = get_loaders(args, patchify=args.patchify,
                                                            batch_size=args.batch_size)
            concepts = [get_concepts(args.concept_name, patchify=args.patchify)]
            num_concepts = [len(concepts[0])]
            num_classes = len(classes)

            binary_inds = get_concept_indicators(args.concept_name) if args.patchify \
                else torch.ones([num_classes, num_concepts[0]])
            binary_inds = binary_inds.to(device)

        os.makedirs(spec_dir, exist_ok=True)

        if not args.eval:
            with open(spec_dir + "args.txt", 'w') as f:
                f.write(str(args))
            writer = SummaryWriter(spec_dir)

        # this needs to be done for both branches
        # if for some reason the texts are not there, compute their embeddings and load
        # else load
        # I can probably integrate that in the loader function
        texts = []
        text_paths = []

        text_dir = get_feature_dir(args)
        if not args.low_level_only:
            text_path = text_dir + '{}_{}_level_text_features.pt'.format(args.concept_name, 'high')
            text_paths.append(text_path)
        if args.patchify:
            text_path = text_dir + '{}_{}_level_text_features.pt'.format(args.concept_name, 'low')
            text_paths.append(text_path)

        # the specific concept set does not exist create it
        for i, text_path in enumerate(text_paths):
            if not os.path.exists(text_path):
                warnings.warn('!!!! Warning !!!!\n The text features do not exist.\n'
                              'Will create a new file\n'
                              '!!!!!!!!!!!!!!!!!')
                # print(concepts[i])
                clip_model, _ = set_clip_model(args)
                compute_and_save_features(args, clip_model.to(device), None, device=device,
                                          only_text=True, patchify=True if args.low_level_only or i == 1
                    else False)
            texts.append(torch.load(text_path).float())

        # get the dimensionality of CLIP through the features in the dataset
            criterion = nn.CrossEntropyLoss().to(device)

        if args.clip_version in ['ViT', 'RN101']:
            feat_emb = 512
        elif 'RN50' == args.clip_version:
            feat_emb = 1024
        elif 'RN101' == args.clip_version:
            feat_emb = 2048
        feat_emb = 512 if 'ViT' in args.clip_version else 1024

        # build the classifiers first
        classifiers = []
        discoverers = []

        # if low_level_only is active, we consider only the low level
        if not args.low_level_only:
            classifiers.append(
                LinearClassifier(num_concepts[0] if not args.only_images else feat_emb, num_classes).to(device))
            if args.discovery and not args.only_images:
                discoverers.append(DiscoveryMechanism(feat_emb, num_concepts[0], prior=prior).to(device))
        if args.patchify:
            classifiers.append(
                LinearClassifier(num_concepts[-1] if not args.only_images else feat_emb, num_classes).to(device))
            if args.discovery and not args.only_images:
                # print("few dims: ", feat_emb, num_concepts[-1])
                discoverers.append(DiscoveryMechanism(feat_emb, num_concepts[-1], prior=prior).to(device))

        cparameters = [c.parameters() for c in classifiers]
        dparameters = [d.parameters() for d in discoverers]
        optimizer = get_optimizer(args, cparameters, dparameters)

        # load the checkpoints and eval
        if args.eval:
            spec_dir = '{}/{}/{}'.format(spec_dir[:12], 'selected_results', spec_dir[13:])
            print(spec_dir)
            sys.exit()
            if not os.path.exists(spec_dir + 'checkpoint_best.pth.tar'):
                continue
            ckpt = torch.load(spec_dir + 'checkpoint_best.pth.tar')

            if not args.low_level_only:
                classifiers[0].load_state_dict(ckpt['state_dict_high'])
                if args.discovery:
                    discoverers[0].load_state_dict(ckpt['disc_state_dict_high'])
            if args.patcify:
                classifiers[-1].load_state_dict(ckpt['state_dict_low'])
                if args.discovery:
                    discoverers[-1].load_state_dict(ckpt['disc_state_dict_low'])


        logger = tlogger.Logger(spec_dir + 'logs', flush_secs=10)

        start_time = time.time()
        num_samples = 1 if not args.discovery else 1

        ######################################################
        #################### MAIN LOOP #######################
        ######################################################
        best_acc_low, best_acc_high = 0., 0.
        for epoch in range(args.epochs):

            # training
            for c in classifiers:
                c.train()
            if args.discovery:
                for d in discoverers:
                    d.train()
            loss, acc_low, acc_high = train(args, train_loader, texts,
                                            classifiers, criterion, optimizer,
                                            epoch, discoverers, binary_inds)

            # validation
            for c in classifiers:
                c.eval()
            if args.discovery:
                for d in discoverers:
                    d.eval()
            val_loss, val_acc_low, val_acc_high, act_perc_low, act_perc_high = validate(args, val_loader, texts,
                                                                                        classifiers, criterion,
                                                                                        discoverers, num_samples,
                                                                                        binary_inds)

            log_stats = 'Epoch: {}, Train Loss: {:.4f}, Train Acc Low: {:.4f}, Train Acc High: {:.4f},' \
                        'Val Loss: {:.4f}, Val Acc Low: {:.4f}, Val Acc High: {:.4f}, ' \
                        'Avg Perc Act Low: {:.4f}, Avg Perc Act High: {:.4f}\n\n'. \
                format(epoch, loss, acc_low, acc_high,
                       val_loss, val_acc_low, val_acc_high, act_perc_low, act_perc_high)

            with open(spec_dir + "log.txt", 'a') as f:
                f.write(log_stats)

            if val_acc_low >= best_acc_low and val_acc_high >= best_acc_high:
                best_acc_low = val_acc_low
                best_acc_high = val_acc_high

                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_high': classifiers[0].state_dict() if not args.low_level_only else None,
                    'state_dict_low': classifiers[-1].state_dict() if args.patchify else None,
                    'disc_state_dict_high': discoverers[
                        0].state_dict() if not args.low_level_only and args.discovery else None,
                    'disc_state_dict_low': discoverers[-1].state_dict() if args.patchify and args.discovery else None,
                    'best_acc1_low': best_acc_low,
                    'best_acc1_high': best_acc_high,
                    'optimizer': optimizer.state_dict(),
                }, is_best=True, filename=spec_dir + 'checkpoint.pth.tar')
                with open(spec_dir + 'best_acc.txt', 'w') as f:
                    f.write('Best acc low: {}, Best acc high: {}'.format(best_acc_low, best_acc_high))

            if epoch % 5 == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict_high': classifiers[0].state_dict() if not args.low_level_only else None,
                    'state_dict_low': classifiers[-1].state_dict() if args.patchify else None,
                    'disc_state_dict_high': discoverers[
                        0].state_dict() if not args.low_level_only and args.discovery else None,
                    'disc_state_dict_low': discoverers[-1].state_dict() if args.patchify and args.discovery else None,
                    'best_acc1_low': best_acc_low,
                    'best_acc1_high': best_acc_high,
                    'optimizer': optimizer.state_dict(),
                }, is_best=False, filename=spec_dir + 'checkpoint.pth.tar')

            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Loss/test', val_loss, epoch)
            writer.add_scalar('Accuracy/train_low', acc_low, epoch)
            writer.add_scalar('Accuracy/train_high', acc_high, epoch)
            writer.add_scalar('Accuracy/test_low', val_acc_low, epoch)
            writer.add_scalar('Accuracy/test_high', val_acc_high, epoch)
            writer.add_scalar('Sparsity/Low', act_perc_low, epoch)
            writer.add_scalar('Sparsity/High', act_perc_high, epoch)
            print('Epoch: {}, Top Validation Acc@1 High {:.3f}, '
                  'Top Validation Acc@1 Low {:.3f}'.format(epoch, best_acc_high, best_acc_low))

        del optimizer, classifiers, discoverers, logger, criterion, cparameters, dparameters, val_loader, train_loader
        # for every exp-freq epochs, update the values of the similarities, save and reload
        # if (epoch +1) % args.exp_freq == 0:
        #    train_loader = update_dataset(args, train_loader, classifier)
