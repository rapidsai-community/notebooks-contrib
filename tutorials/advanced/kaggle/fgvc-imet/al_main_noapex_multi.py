import argparse
import json
import shutil
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import fbeta_score
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
from torch.utils.data import DataLoader

import models_ds as models
#from al_dataset import TrainDataset, N_CLASSES, DATA_ROOT#, TTADataset
#from al_transforms import get_train_transform, get_test_transform, get_test_tta_transform
from py_dataset import TrainDataset, TTADataset, N_CLASSES, DATA_ROOT
from py_transforms import get_train_transform, get_test_transform, get_test_tta_transform

from losses import FocalLoss2d, JointLoss, bce_with_pos_weight, binary_focal_loss, FbetaLoss
from utils import (
    get_learning_rate, set_learning_rate, set_seed,
    write_event, load_model, mean_df, ON_KAGGLE)

#from test_dataset import TTADataset,fivecrop_transform,test_transform

def main():
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('--mode', default='train')
    arg('--run_root', default='../output/')
    arg('--model', default='seresnext50')
    arg('--pretrained', type=int, default=0)
    arg('--batch-size', type=int, default=128)
    arg('--step', type=int, default=1)
    arg('--gamma', type=float, default=0.1)
    arg('--gamma-step', type=int, default=3)
    arg('--workers', type=int, default=2 if ON_KAGGLE else 8)
    arg('--lr', type=float, default=0.0002)
    arg('--warmup-lr', type=float, default=0.01)
    arg('--patience', type=int, default=0)
    arg('--clean', action='store_true')
    arg('--n-epochs', type=int, default=12)
    arg('--epoch-size', type=int)
    arg('--tta', type=int, default=0)
    arg('--debug', action='store_true')
    arg('--limit', type=int)
    arg('--fold', type=int, default=0)
    arg('--multi-gpu', type=int, default=0)
    arg('--size', type=int, default=384)
    arg('--two-stage', type=int, default=1)
    args = parser.parse_args()
    
    set_seed()
    args.run_root = args.run_root  + '/' +  args.model
    run_root = Path(args.run_root)
    folds = pd.read_csv('folds.csv')
    train_root = DATA_ROOT / 'train'
    train_fold = folds[folds['fold'] != args.fold]
    valid_fold = folds[folds['fold'] == args.fold]
    
    if args.limit:
        train_fold = train_fold[:args.limit]
        valid_fold = valid_fold[:args.limit]

    def make_loader(df: pd.DataFrame, image_transform) -> DataLoader:
        return DataLoader(
            TrainDataset(train_root, df, image_transform, debug=args.debug),
            shuffle=True,
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
    
    # define transform
    train_transform = get_train_transform(args.size)
    if args.tta == 0:
        test_transform = get_test_transform(args.size)
        args.tta = 1
    else:
        test_transform = get_test_tta_transform(args.size)
    
    # define loss
    # criterion1 = JointLoss(nn.BCEWithLogitsLoss(), FocalLoss2d(), second_weight=1)
    # criterion2 = JointLoss(nn.BCEWithLogitsLoss(), FocalLoss2d(), second_weight=2)
    criterion = nn.BCEWithLogitsLoss()
    if args.mode == 'train':
        if run_root.exists() and args.clean:
            shutil.rmtree(run_root)
        run_root.mkdir(exist_ok=True, parents=True)
        (run_root / 'params.json').write_text(
            json.dumps(vars(args), indent=4, sort_keys=True))

        training_set = TrainDataset(train_root, train_fold,
                                    transform=train_transform,
                                    over_sample=False,
                                    debug=args.debug)
        validation_set = TrainDataset(train_root, valid_fold,
                                      transform=test_transform,
                                      debug=args.debug)

        print(f'{len(training_set):,} items in train, ', f'{len(validation_set):,} in valid')

        strategy = {
            # 0: {'unfreeze':['layer4','layer3']},
            # 7: {'unfreeze':'layer3'},
            # 9: {'lr': 0.00002},
            # 12: {'unfreeze': ['all']}
            # 10: {'lr':0.00002}
        }

        model = getattr(models, args.model)(num_classes=N_CLASSES, pretrained=True)
        model.freeze()
        model.unfreeze('layer4')
        model.unfreeze('layer3')
        model.cuda()

        optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), args.lr)
        scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.1, verbose=True, mode='max', min_lr=2e-7)
        # scheduler = CyclicLR(optimizer, base_lr=0.00002, max_lr=0.0002, step_size_up=1000, step_size_down=1000, mode='triangular2', cycle_momentum=False)
        #model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

        if args.multi_gpu == 1:
            model = nn.DataParallel(model)

        train_loader = DataLoader(training_set, shuffle=True, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.workers, drop_last=True)
        valid_loader = DataLoader(validation_set, shuffle=False, 
                                  batch_size=args.batch_size, 
                                  num_workers=args.workers)

        train_kwargs = dict(
            args=args,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            train_loader=train_loader,
            valid_loader=valid_loader,
            # use_cuda=use_cuda,
            epoch_length=len(training_set),
            strategy = strategy
        )

        train_kwargs['args'] = args
        #train(n_epochs=args.n_epochs, **train_kwargs)
        if args.two_stage == 1: #switch loss
            print('Change loss to f2 loss')
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), 1e-5)
            scheduler = ReduceLROnPlateau(optimizer, patience=args.patience, factor=0.1, verbose=True, mode='max', min_lr=2e-7)
            train_kwargs['criterion'] = FbetaLoss(beta=2)
            train_kwargs['optimizer'] = optimizer
            train_kwargs['scheduler'] = scheduler
            train(n_epochs=5, **train_kwargs)
        # args.lr/=10

    elif args.mode == 'validate':
        model = getattr(models, args.model)(num_classes=N_CLASSES, pretrained=True)
        valid_loader = DataLoader(
            TrainDataset(train_root, valid_fold,
                         transform=test_transform,
                         debug=args.debug),
                         shuffle=False,
                         batch_size=args.batch_size,
                         num_workers=args.workers,
                         )
        load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=False)
        model.cuda()
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation', ascii=True), args, save_result=True)

    elif args.mode.startswith('predict'):
        model = getattr(models, args.model)(num_classes=N_CLASSES, pretrained=True)
        load_model(model, run_root /('best-model-%d.pt' % args.fold))
        model = model.cuda()
        predict_kwargs = dict(
            batch_size=args.batch_size,
            tta=args.tta,
            workers=args.workers,
            transform=test_transform,
        )
        if args.mode == 'predict_valid':
            predict(model, df=valid_fold, root=train_root,
                    out_path = run_root/ 'valid_fold{}'.format(args.fold),
                    **predict_kwargs)
        elif args.mode == 'predict_test':
            test_root = DATA_ROOT / 'test'
            ss = pd.read_csv(DATA_ROOT / 'sample_submission.csv')
            if args.limit:
                ss = ss[:args.limit]
            predict(model, df=ss, root=test_root,
                    out_path=run_root / 'test_fold{}.h5'.format(args.fold),
                    **predict_kwargs)

    elif args.mode == 'validate_tta':
        model = getattr(models, args.model)(num_classes=N_CLASSES, pretrained=True)
        valid_loader = DataLoader(
            TTADataset(train_root, valid_fold,
                       transform=test_transform,
                       tta=args.tta,
                       valid=True),
            shuffle=False,
            batch_size=args.batch_size,
            num_workers=args.workers
        )
        load_model(model, run_root / ('best-model-%d.pt' % args.fold), multi2single=True)
        model.cuda()
        validation(model, criterion, tqdm.tqdm(valid_loader, desc='Validation', ascii=True), args, save_result=False)
        
def predict(model, root: Path, df: pd.DataFrame, out_path: Path,
            batch_size: int, tta: int, workers: int, transform):
    loader = DataLoader(
        dataset=TTADataset(root, df, transform, tta=tta),
        shuffle=False,
        batch_size=batch_size,
        num_workers=workers,
    )
    model.eval()
    all_outputs, all_ids = [], []
    with torch.no_grad():
        for inputs, ids in tqdm.tqdm(loader, desc='Predict'):
            inputs = inputs.cuda()
            outputs = torch.sigmoid(model(inputs))
            all_outputs.append(outputs.data.cpu().numpy())
            all_ids.extend(ids)
    df = pd.DataFrame(
        data=np.concatenate(all_outputs),
        index=all_ids,
        columns=map(str, range(N_CLASSES)))
    df = mean_df(df)
    df.to_hdf(out_path, 'prob', index_label='id')
    print(f'Saved predictions to {out_path}')


def train(args, model: nn.Module, optimizer, scheduler, criterion, *,
          train_loader, valid_loader, epoch_length,  # use_cuda,
          n_epochs=None, strategy=None) -> bool:
    n_epochs = n_epochs or args.n_epochs

    run_root = Path(args.run_root)
    model_path = run_root / ('model-%d.pt' % args.fold)
    best_model_path = run_root / ('best-model-%d.pt' % args.fold)
    if best_model_path.exists():
        state, best_valid_f2 = load_model(model, best_model_path)
        best_valid_f2 = state['best_valid_loss']
        start_epoch = state['epoch']
        step = state['step']
    else:
        best_valid_f2 = 0
        start_epoch = 0
        step = 0
        best_epoch = 0
    save = lambda ep: torch.save({
        'model': model.state_dict(),
        'epoch': epoch,
        'step': step,
        'best_valid_loss': best_valid_f2
    }, str(model_path))
    #
    report_each = 10000
    log = run_root.joinpath('train-%d.log' % args.fold).open('at', encoding='utf8')

    for epoch in range(start_epoch, start_epoch + n_epochs):
        model.train()

        if epoch in strategy:
            if 'lr' in strategy[epoch]:
                set_learning_rate(optimizer, strategy[epoch]['lr'])
                print('change learning to: %.5f' % strategy[epoch]['lr'])
            if 'unfreeze' in strategy[epoch]:
                for name in strategy[epoch]['unfreeze']:
                    if args.multi_gpu:
                        unfreezed_params = model.module.unfreeze(name)
                    else:
                        unfreezed_params = model.unfreeze(name)
                    print('unfreeze %s' % name)
                    lr = get_learning_rate(optimizer)
                    if len(unfreezed_params)>0:
                        optimizer.add_param_group({'params':unfreezed_params, 'lr':lr})


        lr = get_learning_rate(optimizer)
        tq = tqdm.tqdm(total=epoch_length, ascii=True)
        tq.set_description(f'Epoch {epoch}, lr {lr}')
        losses = []

        mean_loss = 0
        for i, (inputs, targets) in enumerate(train_loader):
            
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # *N_CLASSES  # _reduce_loss
            
            #with amp.scale_loss(loss, optimizer) as scaled_loss:
            #    scaled_loss.backward()
            loss.backward()
            if (i+1) % args.step == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                
                batch_size = inputs.size(0)
                tq.update(batch_size*args.step)
                tq.set_postfix(loss=f'{mean_loss:.5f}')
                losses.append(loss.item())
                mean_loss = np.mean(losses[-report_each:])
            
            if i and i % report_each == 0:
                write_event(log, step, loss=mean_loss)

        write_event(log, step, epoch=epoch, loss=mean_loss)
        tq.close()
        save(epoch + 1)
        valid_metrics = validation(model, criterion, valid_loader, args)
        write_event(log, step, epoch, **valid_metrics)
        current_f2 = valid_metrics['best_f2']
        if scheduler is not None:
            scheduler.step(current_f2)
        if current_f2 > best_valid_f2:
            best_valid_f2 = current_f2
            shutil.copy(str(model_path), str(best_model_path))
            best_epoch = epoch
    return True

def get_score(all_targets, y_pred):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UndefinedMetricWarning)
        return fbeta_score(
            all_targets, y_pred, beta=2, average='samples')
        
def validation(model: nn.Module, criterion, valid_loader, args, save_result = False) -> Dict[str, float]:
    run_root = Path(args.run_root)
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(valid_loader):
            all_targets.append(targets.numpy().copy())

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # *N_CLASSES
            all_losses.append(loss.item())  # _reduce_loss
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    if save_result:
        np.save(run_root / 'prediction.npy', all_predictions)
        np.save(run_root / 'target.npy', all_targets)

    metrics = {}

    # culture_probs = all_predictions[:, 0:398]
    # tag_probs = all_predictions[:, 398:]
    # tag_argsorted = tag_probs.argsort(axis=1)

    argsorted = all_predictions.argsort(axis=1)
    metrics['best_f2'] = 0
    for threshold in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
        score = get_score(all_targets, 
                          binarize_prediction(all_predictions, threshold, argsorted))  # tag_argsorted=tag_argsorted, culture_probs=culture_probs
        metrics[f'valid_f2_th_{threshold:.2f}'] = score
        if score > metrics['best_f2']:
            metrics['best_f2'] = score
    metrics['valid_loss'] = np.mean(all_losses)
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.4f}')
    print(' | '.join(to_print))
    return metrics

def validation_TTA(model: nn.Module, criterion, valid_loader, args, save_result = False) -> Dict[str, float]:
    run_root = Path(args.run_root)
    model.eval()
    all_losses, all_predictions, all_targets = [], [], []
    with torch.no_grad():
        for inputs, targets in valid_loader:
            all_targets.append(targets.numpy().copy())
            bs, ncrops, c, h, w = inputs.size()
            inputs = inputs.view(-1, c, h, w)

            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = model(inputs)

            outputs = outputs.view(bs, ncrops, -1).mean(1)

            loss = criterion(outputs, targets)  # *N_CLASSES
            all_losses.append(loss.item())  # _reduce_loss
            predictions = torch.sigmoid(outputs)
            all_predictions.append(predictions.cpu().numpy())
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)

    if save_result:
        np.save(run_root / 'prediction.npy', all_predictions)
        np.save(run_root / 'target.npy', all_targets)

    metrics = {}
    argsorted = all_predictions.argsort(axis=1)
    metrics['best_f2'] = 0
    for threshold in [0.1, 0.15, 0.2, 0.25]:
        score = get_score(all_targets, binarize_prediction(all_predictions, threshold,
                                              argsorted))  # tag_argsorted=tag_argsorted, culture_probs=culture_probs
        metrics[f'valid_f2_th_{threshold:.2f}'] = score
        if score > metrics['best_f2']:
            metrics['best_f2'] = score
    metrics['valid_loss'] = np.mean(all_losses)
    to_print = []
    for idx, (k, v) in enumerate(sorted(metrics.items(), key=lambda kv: -kv[1])):
        to_print.append(f'{k} {v:.5f}')
    print(' | '.join(to_print))
    return metrics

def binarize_prediction(probabilities, threshold: float, argsorted=None,
                        min_labels=1, max_labels=10):
    """ Return matrix of 0/1 predictions, same shape as probabilities.
    """
    assert probabilities.shape[1] == N_CLASSES
    if argsorted is None:
        argsorted = probabilities.argsort(axis=1)
    max_mask = _make_mask(argsorted, max_labels)
    min_mask = _make_mask(argsorted, min_labels)
    prob_mask = probabilities > threshold
    return (max_mask & prob_mask) | min_mask

def _make_mask(argsorted, top_n: int):
    mask = np.zeros_like(argsorted, dtype=np.uint8)
    col_indices = argsorted[:, -top_n:].reshape(-1)
    row_indices = [i // top_n for i in range(len(col_indices))]
    mask[row_indices, col_indices] = 1
    return mask

def _reduce_loss(loss):
    return loss.sum() / loss.shape[0]


if __name__ == '__main__':
    main()
    # binarize_prediction(np.array([[0.1, 0.2, 0.3],[0.1, 0.2, 0.3]]), 0.1)
