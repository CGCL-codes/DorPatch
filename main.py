from utils import *
import argparse
from attack import DorPatch
from defenses.PatchCleanser import PatchCleanser, MaskWindow
from tqdm import tqdm
import pickle

parser = argparse.ArgumentParser(
    description='set parameters for patch generation')

parser.add_argument('--device', default='0', type=str, metavar='DEVICE',
                    help='gpu device id')
parser.add_argument('--dataset', '-d', default='imagenet', type=str, metavar='DATASET',
                    help='dataset', choices=['cifar10', 'imagenet', 'cifar100'])
parser.add_argument('--data_dir', default='/home/data/data',
                    help='path to dataset')
parser.add_argument('--model_dir', default='pretrained_models/',
                    help='path to model')
parser.add_argument('--base_arch', '-ba', metavar='BARCH', default='resnetv2', choices=['resnetv2'],
                    help='base model architecture for patch generation (default: resnetv2)')
parser.add_argument('--targeted', '-t',  action='store_true',
                    help='targeted attack or not')
parser.add_argument('--patch_budget', default=0.12,
                    type=float, help='patch budget')
parser.add_argument('--attack', '-a', default='DorPatch', type=str, metavar='ATTACK',
                    help='atttack method', choices=['DorPatch'])
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 64)') 
parser.add_argument('-e', '--epsilon', default=4., type=float,
                    metavar='E', help='epsilon to bound the perturbation (l2 norm)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
# settings for DorPatch
parser.add_argument('--num_patch', default=-1, type=int,
                    help='number of patches (default: -1 as unconstrained)')
parser.add_argument('--dropout',  default=2, type=int,
                    help='using how many rounds of image dropout (for robustness to occlusion)')
parser.add_argument('--density', default=1e-3, type=float,
                    help='the coeff of density regularization (for distributed property) or not')
parser.add_argument('--structured', default=1e-3, type=float,
                    help='the coeff of structured loss')


def main(args):
    set_device(args.device)
    set_random_seed()

    result_dir = generate_saving_path(vars(args).copy())    

    # prepare the model and dataset
    model = get_model(args.dataset, args.base_arch, args.model_dir)
    model = NormModel(model, get_normalize(args.dataset, args.base_arch))
    model = torch.nn.DataParallel(model)
    model.cuda()
    model.eval()

    dataloader = get_dataset(
        args.dataset, data_dir=args.data_dir, batch_size=args.batch_size)

    attack = DorPatch()
    defense = [PatchCleanser(MaskWindow(224, r, 1), model) for r in [0.015, 0.03, 0.06, 0.12]]

    with torch.no_grad():

        # # evaluate the classifier first
        # preds_list = []
        # gt_list = []
        # for i, (x, y) in tqdm(enumerate(dataloader)):
        #     preds_list.append(model(x.cuda()).argmax(-1).cpu().numpy())
        #     gt_list.append(y.cpu().numpy())
        # print("clean accuracy on %s is %.2f" % (args.dataset, (np.concatenate(preds_list) == np.concatenate(gt_list)).mean()*100))
        
        
        # ========== generate the DorPatch samples ==========
        # store the evaluation results
        target_list = []
        preds_list = []
        y_list = []
        preds_adv_list = []
        records = []

        for i, (x, y) in tqdm(enumerate(dataloader)):
            
            if i == 10:  # set the number larger/smaller if you want to test on more/less images
                break

            x = x.cuda()
            y = y.cuda()

            # filter out the incorrectedly classified images
            logits = model(x)
            preds = logits.argmax(-1)
            correct = (preds == y)
            if correct.sum() == 0:
                continue
            x = x[correct]
            y = y[correct]
            preds = preds[correct]
            logits = logits[correct]

            if os.path.exists(os.path.join(result_dir, "adv_mask_%d.pt" % i)):
                # load the generated patch (mask and pattern) if exists
                adv_mask = torch.load(os.path.join(
                    result_dir, "adv_mask_%d.pt" % i))
                adv_pattern = torch.load(os.path.join(
                    result_dir, "adv_pattern_%d.pt" % i))
                
                if args.targeted:
                    # recover target label from stage 0
                    dir_0 = os.path.join(*result_dir.split('/')[:-1])
                    adv_mask_0 = torch.load(os.path.join(
                        dir_0, "adv_mask_%d.pt" % i))
                    adv_pattern_0 = torch.load(os.path.join(
                        dir_0, "adv_pattern_%d.pt" % i))
                    delta_x_0 = clip(adv_mask_0, adv_pattern_0, x, args.epsilon)           
                    adv_x_0 = x + delta_x_0
                    target_list.append(model(adv_x_0).argmax(-1).cpu().numpy())
                    assert (target_list[-1] != y.cpu().numpy()).all()
            else:
                if args.targeted:
                    # set to a random target for targerted attack
                    target = torch.randint(0, NUM_CLASSES_DICT[args.dataset], x.shape[:1]).cuda()
                    assert (target != y).all()
                    target_list.append(target.cpu().numpy())

                model_tgt = model

                with torch.enable_grad():
                    adv_mask, adv_pattern = attack.generate(
                        model_tgt, x, args.patch_budget, NUM_CLASSES_DICT[args.dataset], targeted=args.targeted,
                        y = target if args.targeted else None,
                        lr=args.lr, num_patch=args.num_patch,
                        dropout=args.dropout, density=args.density, structured=args.structured,
                        save_dir=result_dir, batch_id=i, eps=args.epsilon)
                torch.save(adv_mask, os.path.join(
                    result_dir, "adv_mask_%d.pt" % i))
                torch.save(adv_pattern, os.path.join(
                    result_dir, "adv_pattern_%d.pt" % i))   
                
            delta_x = clip(adv_mask, adv_pattern, x, args.epsilon)
            adv_x = x + delta_x

            # PatchCleanser
            pc_path = os.path.join(result_dir, "adv_PC_%d.pt" % i)
            if os.path.exists(pc_path):
                with open(pc_path, 'rb') as f:
                    records_batch = pickle.load(f)
            else:
                records_batch = []
                for img in adv_x:
                    records_batch.append([d.robust_predict(img, True) for d in defense])
                with open(pc_path, 'wb') as f:
                    pickle.dump(records_batch, f)

            preds_list.append(preds.cpu().numpy())
            y_list.append(y.cpu().numpy())
            
            adv_logits = model(adv_x)
            preds_adv_list.append(adv_logits.argmax(-1).cpu().numpy())
            records += records_batch

        if args.targeted:
            target_list = np.concatenate(target_list)
        preds_list = np.concatenate(preds_list)
        y_list = np.concatenate(y_list)
        preds_adv_list = np.concatenate(preds_adv_list)

        acc_clean = (preds_list == y_list).mean()*100
        acc_robust = (preds_adv_list == y_list).mean()*100

        for i, d in enumerate(defense):
            d.collect([r[i] for r in records])
        pred_prov = [d.result.predictions for d in defense]
        certifiable = [d.result.certifications for d in defense]
        acc_PC = [(p == y_list).mean() * 100 for p in pred_prov]
        certified_acc_PC = [((p == y_list) &
                            c).mean() * 100 for p, c in zip(pred_prov, certifiable)]
        if args.targeted:
            #     ASR = (preds_adv_list == target_list).mean()*100
            #     ASR_pc = [(p == target_list).mean() * 100 for p in pred_prov]
            certified_asr_PC = [((p == target_list) & c).mean() * 100 for p, c in zip(pred_prov, certifiable)]
        else:

            certified_asr_PC = [((p != y_list) & c).mean() * 100 for p, c in zip(pred_prov, certifiable)]
        
        print("clean accuracy: {:.2f}%, robust accuracy:{:.2f}%, acc@PC:{:s}%, certified_ACC@PC:{:s}%, certified_ASR@PC:{:s}%".format(
            acc_clean, acc_robust, convert_float_list_to_str(acc_PC), convert_float_list_to_str(certified_acc_PC), convert_float_list_to_str(certified_asr_PC)))


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
