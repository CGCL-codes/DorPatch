import torch
import numpy as np
from tqdm import tqdm
import os
from defenses.PatchCleanser import MaskWindow
from torchvision.utils import save_image
import torchvision
from utils import clip

class CW_loss():
    def __init__(self, num_classes, targeted=False, confidence=0):
        self.num_classes = num_classes
        self.targeted = targeted
        self.confidence = confidence

    def __call__(self, logits, y):
        y_onehot = torch.nn.functional.one_hot(y, self.num_classes)
        real = (logits * y_onehot).sum(1)
        other = ((1. - y_onehot) * logits - (y_onehot * 1e4)).max(1)[0]
        if self.targeted:
            return torch.clamp(self.confidence + other - real, min=0.)
        else:
            return torch.clamp(self.confidence + real - other, min=0.)

def get_mask_set(img_size, dropout_size, dropout):
    # prepare the mask set for image dropout
    mask_window = MaskWindow(img_size, dropout_size)
    if dropout == 1:
        return mask_window.mask_set
    elif dropout == 2:
        return mask_window.double_mask_set

def local_variance(x):
    # local variance of an image
    grad_left_right = x.clone().detach()
    grad_left_right[:, :, :, :-1].sub_(x[:, :, :, 1:]).abs_()
    grad_up_down = x.clone().detach()
    grad_up_down[:, :, :-1, :].sub_(x[:, :, 1:, :]).abs_()
    return (grad_left_right + grad_up_down), grad_left_right, grad_up_down

def min_var_weighted_variance(x):
    # variant of variation loss
    # regularize the direction with larger image gradient
    local_var, grad_left_right, grad_up_down = local_variance(x)
    return local_var * torch.where(grad_left_right > grad_up_down, grad_up_down, grad_left_right)

class DorPatch(object):
    def __init__(self):
        pass

    def generate(self, model, x, patch_budget, n_classes, save_dir, batch_id, y=None, targeted=False,
                lr=1e-2, confidence=1e-1, clip_min=0, clip_max=1, max_iterations=5000, basic_unit=7,
                selection='topk', dropout=2, sampling_size=128, density=1e-3, structured=1e-3, eps=4., dual=False, **kwargs):
        if dropout == 0:
            sampling_size = 0

        self.criterion = CW_loss(n_classes, targeted, confidence)  # adversarial loss
        x_shape = tuple(x.shape)
        adv_mask = torch.rand([x_shape[0], 1, x_shape[2], x_shape[3]]).cuda()
        adv_pattern = torch.rand(x_shape).cuda()
        adv_mask.requires_grad = True
        adv_pattern.requires_grad = True
        adv_mask_best_np = np.zeros(adv_mask.shape, dtype=np.float32)
        adv_pattern_best_np = np.zeros(adv_pattern.shape, dtype=np.float32)
        patience = 200

        if y is None:  # set current preds as ground truths if not provided
            with torch.no_grad():
                y = model(x).argmax(-1)

        # for group-wise sparsity
        conv_group = torch.nn.Conv2d(
            1, 1, basic_unit, stride=basic_unit, bias=False, padding=0).cuda()
        conv_group.weight.data[:, :, :, :] = 1
        
        # for density regularization
        window_size = int(x_shape[-1] // 8)
        conv_density = torch.nn.Conv2d(
            1, 1, window_size, stride=window_size, bias=False, padding=0).cuda()
        conv_density.weight.data[:, :, :, :] = 1

        # expected to survive different dropout sizes of Patch
        dropout_size = [0.015, 0.03, 0.06, 0.12]
        mask_set_universe = [get_mask_set(x_shape[-1], ds, dropout) for ds in dropout_size]
        mask_set_universe = torch.concat(mask_set_universe, dim=0)

        coeff_group_lasso = 1e-5
        scale_up = 1.2
        scale_down = np.sqrt(scale_up ** 3)
        certifiable = False

        n_mask = mask_set_universe.size(0)
        if n_mask < sampling_size:
            sampling_size = n_mask
        sampling_choices = np.arange(n_mask)
        failed_idxs = np.array([], dtype=np.int32)

        y = y[:, None].expand((1, sampling_size)).view((-1))
        
        local_var_x = local_variance(x)[0].mean(1)  # for the item to weight for structual loss

        # save the importance map in the mother dir for re-using to generate patch @different budget
        dir_0 = os.path.join(*save_dir.split('/')[:-1])


        def set_target(preds_adv, label):
            # convert untargeted attack to targeted attack
            # set to targeted to the major mis-classified category (easiest)
            # for label consistency for certification
            incorrect = (preds_adv != label)
            preds_adv_ = preds_adv[incorrect]
            if len(preds_adv_) > 0:
                self.criterion = CW_loss(
                    n_classes, True, confidence)
                if (len(preds_adv_)) > 1:
                    # return the majority class ! other than original label
                    target = preds_adv_.view((x_shape[0], -1)).mode(1)[0]
                else:
                    target = preds_adv_.detach().clone()
                return target[:, None].expand((1, sampling_size,)).view((-1))
            else:
                return label

        for stage in range(2):
            # Stage 0: generate importance map (using group lasso)
            # Stage 1: optimize patch content (pattern)
            print('============= Stage %d =============' % stage)
            # decay learning rate when loss doesn't decay for patience steps
            lr_current = torch.ones(x_shape[0]).cuda() * lr
            loss_best = torch.ones(x_shape[0]).cuda() * torch.inf
            not_decay = torch.zeros(x_shape[0], dtype=torch.int).cuda()
            num_failure = torch.inf

            if stage == 0 and os.path.exists(os.path.join(dir_0, "adv_mask_%d.pt" % batch_id)):
                # load the mask and pattern for the first stage if exists
                adv_mask_best = torch.load(os.path.join(
                    dir_0, "adv_mask_%d.pt" % batch_id))
                adv_pattern_best = torch.load(os.path.join(
                    dir_0, "adv_pattern_%d.pt" % batch_id))

                continue

            if stage == 1:
                with torch.no_grad():
                    adv_mask_best_np = adv_mask_best.cpu().numpy()
                    adv_pattern_best_np = adv_pattern_best.cpu().numpy()
                    # set the target
                    delta_x = clip(adv_mask_best, adv_pattern_best, x, eps)
                    adv_x = delta_x + x

                    if not targeted:
                        adv_logits = model(adv_x)
                        preds_adv = adv_logits.argmax(-1)
                        targeted = True
                        y = set_target(preds_adv)

                    # use the merged content as initial for stage 1
                    adv_pattern_best.data = adv_x.data  
                    adv_pattern_best_np = adv_x.cpu().numpy()

                    adv_mask.data = self.patch_selection(
                        adv_mask_best, patch_budget, basic_unit, selection).data
                    adv_mask_best_np = adv_mask.cpu().numpy()
                    adv_mask.requires_grad = False  # fix the mask for stage 1
                    adv_pattern.data = adv_pattern_best.data

            for i in (range(max_iterations)):

                if stage ==0 and i == 500 and not targeted:
                    targeted = True
                    preds_adv = adv_logits.argmax(-1)
                    y_new = set_target(preds_adv, y)
                    if (y_new != y).any():
                        targeted = True
                        y = y_new
                        print(">> switch to targeted attack to category {:3d} at iteration: {:4d}".format(y[0].item(), i))
                    lr_current = torch.ones(x_shape[0]).cuda() * lr
                    loss_best = torch.ones(x_shape[0]).cuda() * torch.inf
                    not_decay = torch.zeros(x_shape[0], dtype=torch.int).cuda()
                    num_failure = torch.inf
                    failed_idxs = self.collect_failure(adv_x, y, mask_set_universe,
                                    targeted, model, batch_size=sampling_size)

                delta_x = clip(adv_mask, adv_pattern, x, eps)
                adv_x = delta_x + x
                
                if i % 100 == 0:
                    # update global failure indexes periodically
                    failed_idxs = self.collect_failure(adv_x, y, mask_set_universe,
                                                       targeted, model,batch_size=sampling_size)
                    
                # sampling the mask set
                n_form_failure = 0 if i < 1000 else min(len(failed_idxs), sampling_size//2)
                n_form_universe = sampling_size - n_form_failure

                sampling_idxs = []
                if n_form_failure > 0:
                    sampling_idxs.append(np.random.choice(
                        failed_idxs, n_form_failure, replace=False))
                if n_form_universe > 0:
                    sampling_idxs.append(np.random.choice(
                        sampling_choices, n_form_universe, replace=False))
                sampling_idxs = np.concatenate(sampling_idxs)
                mask_set = mask_set_universe[sampling_idxs]

                adv_x_masked = adv_x[:, None,] * mask_set + 0.5 * ~mask_set

                if dual:
                    sampling_idxs_dual = []
                    if n_form_failure > 0:
                        sampling_idxs_dual.append(np.random.choice(
                            failed_idxs, n_form_failure, replace=False))
                    if n_form_universe > 0:
                        sampling_idxs_dual.append(np.random.choice(
                            sampling_choices, n_form_universe, replace=False))
                    sampling_idxs_dual = np.concatenate(sampling_idxs_dual)
                    mask_set_dual = mask_set_universe[sampling_idxs_dual]
                    adv_x_masked = adv_x_masked * mask_set_dual + 0.5 * ~mask_set_dual

                adv_x_masked = adv_x_masked.view((-1, ) + adv_x_masked.shape[2:])

                adv_logits = model(adv_x_masked)

                loss_adv = self.criterion(adv_logits, y).view(
                    (x_shape[0], sampling_size))

                min_var_adv_x = min_var_weighted_variance(adv_x).mean(1)
                loss_struc = torch.mean(min_var_adv_x/(local_var_x+1e-5), (1, 2))  # avoid zero division

                loss = loss_adv.mean(1)

                if structured != 0:  #  and stage == 1
                    loss.add_(structured * loss_struc)

                if stage == 0:
                    # for being distributed
                    loss_density = conv_density(adv_mask).view((x_shape[0], -1)).var(1)

                    if density != 0:
                        loss.add_(density * loss_density)

                    # for group sparsity (physical realizablity)
                    group_lasso = basic_unit * conv_group(
                        adv_mask**2).sqrt().sum((1, 2, 3))
                    loss.add_(coeff_group_lasso * group_lasso)

                loss.sum().backward()

                with torch.no_grad():
                    if stage == 0:
                        loss_target = group_lasso  # aim to produce the best group sparsity
                    else:
                        loss_target = loss_struc   # aim to produce the best structural 

                    attack_success = (loss_adv < 1e-1)

                    mask_success = attack_success.all(0)

                    new_successes = sampling_idxs[:n_form_failure][
                            (mask_success[:n_form_failure]).cpu().numpy()]
                    if len(new_successes) > 0:
                        failed_idxs = np.setdiff1d(failed_idxs, new_successes).tolist()
                    new_failures = sampling_idxs[n_form_failure:][
                            (~mask_success[n_form_failure:]).cpu().numpy()]
                    if len(new_failures) > 0:
                        failed_idxs.extend(new_failures)
                        failed_idxs = np.unique(failed_idxs).tolist()

                    attack_success = attack_success.all(1)
                    certifiable = (len(failed_idxs) == 0)

                    if (len(failed_idxs) < num_failure):
                        loss_best.data[:] = torch.inf
                    certify_better =  (len(failed_idxs) <= num_failure) 
                    loss_decay = certify_better & ((loss_target - loss_best) < -1e-3)
                    
                    # record the best result and decay learning rate
                    save_best = loss_decay
                    isnt_decay = ~loss_decay
                    
                    if save_best.any():
                        num_failure = len(failed_idxs)
                        loss_best[save_best] = loss_target[save_best]
                    save_best_np = save_best.cpu().numpy()

                    if stage == 0:
                        adv_mask_best_np[save_best_np] = adv_mask[save_best].cpu().numpy()

                    adv_pattern_best_np[save_best_np] = adv_pattern[save_best].cpu().numpy()
                    not_decay[loss_decay] = 0
                    not_decay[isnt_decay] += 1
                    early_stop = (not_decay > patience)

                    if stage == 0 and i>200:
                        if attack_success and certifiable:
                            coeff_group_lasso *= scale_up
                        else:
                            coeff_group_lasso /= scale_down
                    else:
                        if attack_success and certifiable:
                            structured *= scale_up
                        else:
                            structured /= scale_down

                    if early_stop.any():
                        lr_current[early_stop] *= 0.1
                        lr_current.clip_(min=.1/256.)
                        not_decay[early_stop] = 0


                    if (lr_current < 1e-3).all():
                        print("early stop at iteration: {:4d}".format(i))
                        if np.isinf(loss_best.item()):
                            adv_mask_best_np = adv_mask.cpu().numpy()
                            adv_pattern_best_np = adv_pattern.cpu().numpy()       
                        break

                    if i % 20 == 0:
                        preds = adv_logits.argmax(-1)
                        acc = (preds == y).sum().item() / \
                            (x_shape[0] * sampling_size) * 100
                        info_string = "iteration: {:4d}, accuracy: {:.2f}, loss: {:.2f}, adv: {:.2f}, l2 norm: {:.2f}, structural: {:.2f}"
                        l2_norm = torch.norm(delta_x, p=2, dim=(1, 2, 3))
                        info = (i, acc, loss.mean().item(),
                                loss_adv.mean().item(), l2_norm.mean().item(), loss_struc.mean().item())
                        if stage == 0:
                            info_string += ", group lasso: {:.2f}, density: {:.2f}"
                            info += (group_lasso.mean().item(),
                                     loss_density.mean().item())
                        print(info_string.format(*info))

                    # update the mask and pattern with gradient descent
                    if stage == 0:
                        adv_mask.sub_(
                            lr_current[:, None, None, None] * adv_mask.grad.sign())
                        adv_mask.clip_(clip_min, clip_max)
                        adv_mask.grad.zero_()

                    adv_pattern.sub_(
                        lr_current[:, None, None, None] * adv_pattern.grad.sign())
                    adv_pattern.clip_(clip_min, clip_max)
                    adv_pattern.grad.zero_()

            if np.isinf(loss_best.item()):  # no best saved then return last
                adv_mask_best_np = adv_mask.cpu().numpy()
                adv_pattern_best_np = adv_pattern.cpu().numpy()            
            
            if stage == 0:
                with torch.no_grad():
                    # save the mask and pattern for the first stage
                    adv_mask_best = torch.from_numpy(adv_mask_best_np).cuda()
                    adv_pattern_best = torch.from_numpy(adv_pattern_best_np).cuda()
                    torch.save(adv_mask_best, os.path.join(
                        dir_0, "adv_mask_%d.pt" % batch_id))
                    torch.save(adv_pattern_best, os.path.join(
                        dir_0, "adv_pattern_%d.pt" % batch_id))
                if not targeted:
                    preds_adv = adv_logits.argmax(-1)
                    y = set_target(preds_adv)

        return torch.from_numpy(adv_mask_best_np).cuda(), torch.from_numpy(adv_pattern_best_np).cuda()

    def patch_selection(self, mask, patch_budget, basic_unit=7, selection='topk'):
        # generate patch location from importance map (grouped)
        conv_group = torch.nn.Conv2d(
            1, 1, basic_unit, stride=basic_unit, bias=False, padding=0).cuda()
        conv_group.weight.data[:, :, :, :] = 1
        conv_group.weight.requires_grad = False
        group_importance = conv_group(mask)  # window sum of mask
        num_group = np.floor((mask.shape[2] * mask.shape[3] *
                              patch_budget) / (basic_unit**2)).astype(int)
        if selection == 'topk':
            group_importance_flatten = group_importance.view(
                (mask.shape[0], -1))
            value_topk, idx_topk = group_importance_flatten.topk(num_group)

            selected_group = torch.zeros_like(group_importance_flatten)
            for sg, value, idx in zip(selected_group, value_topk, idx_topk):
                sg[idx[value>0]] = 1
            mask_selected = selected_group.view(group_importance.shape).repeat_interleave(
                basic_unit, dim=2).repeat_interleave(basic_unit, dim=3)
        return mask_selected

    def collect_failure(self, adv_x, y, mask_set_universe, targeted, model, batch_size=128, transforms=None):
        n_mask = mask_set_universe.size(0)
        # prepare failed mask idxs for sampling
        failed_idxs = []
        with torch.no_grad():
            for j in range(np.ceil(n_mask / batch_size).astype(int)):
                mask_set = mask_set_universe[j*batch_size: min(
                    (j+1)*batch_size, n_mask)]
                adv_x_masked = adv_x[:, None,] * \
                    mask_set + 0.5 * ~mask_set
                adv_x_masked = adv_x_masked.view((-1, ) + adv_x.shape[1:])
                if transforms is not None:
                    adv_x_masked = transforms(adv_x_masked)
                adv_preds = model(adv_x_masked).argmax(-1)
                failed_idx = (adv_preds == y.view((adv_x.shape[0], batch_size))[
                    :, :mask_set.size(0)].view(-1))
                if targeted:
                    failed_idx = ~failed_idx
                failed_idx = failed_idx.nonzero().view(-1) % batch_size + (j*batch_size)
                failed_idxs.append(failed_idx.unique())  # for multiple inputs, remove the repeated idxs
            failed_idxs = torch.cat(failed_idxs, 0).cpu().numpy().tolist()
            print(">> %d failures collected!" % len(failed_idxs))
        return failed_idxs
