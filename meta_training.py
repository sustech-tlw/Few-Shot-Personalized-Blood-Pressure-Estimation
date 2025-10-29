import torch
import time
from utils import AverageMeter, ProgressMeter, regression_metric, log_and_print, norm, log_and_print_gridsim

def training_one_epoch_meta(data_loader, model, optimizer, epoch, args):
    task_time = AverageMeter('Time', ':6.3f')
    MSE_loss = AverageMeter('MSE_Loss', ':.3f')
    classify_loss = AverageMeter('classify_Loss', ':.3f')
    classify_acc = AverageMeter('classify_acc', ':.3f')
    auxiliary_loss = AverageMeter('auxiliary_loss', ':.3f')
    progress = ProgressMeter(
        len(data_loader),
        [task_time, MSE_loss, classify_loss, classify_acc, auxiliary_loss],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()
    Epoch_preds_support = []
    Epoch_Labels_support = []
    Epoch_preds_query = []
    Epoch_Labels_query = []
    for batch_i, (Signal_batch, Label_batch) in enumerate(data_loader):

        task_losses = []
        task_auxiliary_losses = []
        for person_i in range(len(Signal_batch)):
            start = time.time()
            Signal = Signal_batch[person_i]
            Label = Label_batch[person_i]
            if model.channel_num == 1:
                Signal = Signal[:, 1:2, :]
            support_set_x, support_set_y, query_set_x, query_set_y = data_label_split(Signal, Label, support_num=args.k_shot, query_num=args.k_query, shuffle=False)

            model_params = list(model.parameters())
            frozen_weights = []
            fast_weights = []
            frozen_order = []
            fast_order = []
            for i, p in enumerate(model_params):
                if p.requires_grad:
                    fast_weights.append(p)  # save original weight
                    fast_order.append(i)
                if not p.requires_grad:
                    frozen_weights.append(p)  # save  weight
                    frozen_order.append(i)

            if args.gpu is not None:
                support_set_x = support_set_x.cuda(args.gpu, non_blocking=True).float()
                support_set_y = support_set_y.cuda(args.gpu, non_blocking=True).float()
                query_set_x = query_set_x.cuda(args.gpu, non_blocking=True).float()
                query_set_y = query_set_y.cuda(args.gpu, non_blocking=True).float()

            # inner loops
            for _ in range(1):
                weights = frozen_weights + fast_weights
                order = frozen_order + fast_order
                weights = restore_original_order(weights, order)
                result = forward_with_params(model, support_set_x, weights)
                pred_support = result['pred']
                loss_MSE_support = ((pred_support - support_set_y) ** 2).mean()
                if model.auxiliary_task:
                    regression_support = result['regression']
                    support_set_x_norm = norm(support_set_x)
                    target_support = support_set_x_norm.reshape([regression_support.shape[0], support_set_x_norm.shape[-1]]).reshape(regression_support.shape)
                    loss_auxiliary_support = ((regression_support - target_support) ** 2).mean()
                    loss_support = loss_MSE_support + loss_auxiliary_support
                else:
                    loss_support = loss_MSE_support

                Epoch_Labels_support.append(support_set_y.detach().cpu())
                Epoch_preds_support.append(pred_support.detach().cpu())
                grads = torch.autograd.grad(loss_support, fast_weights, create_graph=True)
                fast_weights = [w - args.meta_inner_lr * g for w, g in zip(fast_weights, grads)]

            adapt_weight = frozen_weights + fast_weights
            order = frozen_order + fast_order
            adapt_weight = restore_original_order(adapt_weight, order)
            result = forward_with_params(model, query_set_x, adapt_weight)
            pred_query = result['pred']
            loss_MSE_query = ((pred_query - query_set_y) ** 2).mean()
            if model.auxiliary_task:
                regression_query = result['regression']
                query_set_x_norm = norm(query_set_x)
                target_query = query_set_x_norm.reshape(
                    [regression_query.shape[0], query_set_x_norm.shape[-1]]).reshape(regression_query.shape)
                loss_auxiliary_query = ((regression_query - target_query) ** 2).mean()
                loss_query = loss_MSE_query
            else:
                loss_query = loss_MSE_query

            Epoch_Labels_query.append(query_set_y.detach().cpu())
            Epoch_preds_query.append(pred_query.detach().cpu())
            task_losses.append(loss_query)
            if model.auxiliary_task:
                task_auxiliary_losses.append(loss_auxiliary_query)


            end = time.time()
            task_time.update(end - start)
            MSE_loss.update(loss_MSE_query.item(), query_set_y.shape[0])
            if model.auxiliary_task:
                auxiliary_loss.update(loss_auxiliary_query.item(), target_query.shape[0])

        progress.display(batch_i)
        if model.auxiliary_task:
            meta_loss = sum(task_losses) / len(task_losses)
            aux_loss = sum(task_auxiliary_losses) / len(task_auxiliary_losses)
            grads_meta = torch.autograd.grad(meta_loss, model.parameters(), retain_graph=True)
            grads_aux = torch.autograd.grad(aux_loss, model.parameters(), retain_graph=True)
            grad_meta_vec = grads_to_vec(grads_meta)
            grad_aux_vec = grads_to_vec(grads_aux)
            cos_sim = torch.nn.functional.cosine_similarity(grad_meta_vec, grad_aux_vec, dim=0)
            log_and_print_gridsim(cos_sim, epoch + batch_i / len(data_loader), save_path=args.save_path, save_name='cos_similarity_train')
            optimizer.zero_grad()
            total_loss = meta_loss + aux_loss
            total_loss.backward()
            optimizer.step()
        else:
            meta_loss = sum(task_losses) / len(task_losses)
            optimizer.zero_grad()
            meta_loss.backward()
            optimizer.step()
        print(f'update parameters {batch_i}')

    Epoch_Labels_support = torch.cat(Epoch_Labels_support, dim=0).squeeze()
    Epoch_preds_support = torch.cat(Epoch_preds_support, dim=0).squeeze()
    Epoch_Labels_query = torch.cat(Epoch_Labels_query, dim=0).squeeze()
    Epoch_preds_query = torch.cat(Epoch_preds_query, dim=0).squeeze()
    metric_epoch_support = regression_metric(Epoch_preds_support.numpy(), Epoch_Labels_support.numpy())
    metric_epoch_query = regression_metric(Epoch_preds_query.numpy(),
                                             Epoch_Labels_query.numpy())
    log_and_print(metric_epoch_support, epoch, args.save_path, 'training_support')
    log_and_print(metric_epoch_query, epoch, args.save_path, 'training_query')
    return


def val_one_epoch_meta(data_loader, model, epoch, args, save_name='validating'):
    task_time = AverageMeter('Time', ':6.3f')
    MSE_loss = AverageMeter('MSE_Loss', ':.3f')
    classify_loss = AverageMeter('classify_Loss', ':.3f')
    classify_acc = AverageMeter('classify_acc', ':.3f')
    auxiliary_loss = AverageMeter('auxiliary_loss', ':.3f')
    progress = ProgressMeter(
        len(data_loader),
        [task_time, MSE_loss, classify_loss, classify_acc, auxiliary_loss],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    Epoch_preds_support = []
    Epoch_Labels_support = []
    Epoch_preds_query = []
    Epoch_Labels_query = []

    for batch_i, (Signal_batch, Label_batch) in enumerate(data_loader):
        task_losses = []

        for person_i in range(len(Signal_batch)):
            start = time.time()
            Signal = Signal_batch[person_i]
            Label = Label_batch[person_i]
            if model.channel_num == 1 and Signal.shape[1] > 1:
                Signal = Signal[:, 1:2, :]
            support_set_x, support_set_y, query_set_x, query_set_y = data_label_split(Signal, Label, support_num=args.k_shot, query_num=args.k_query, shuffle=False)

            if args.gpu is not None:
                support_set_x = support_set_x.cuda(args.gpu, non_blocking=True).float()
                support_set_y = support_set_y.cuda(args.gpu, non_blocking=True).float()
                query_set_x = query_set_x.cuda(args.gpu, non_blocking=True).float()
                query_set_y = query_set_y.cuda(args.gpu, non_blocking=True).float()

            model_params = list(model.parameters())
            frozen_weights = []
            fast_weights = []
            frozen_order = []
            fast_order = []
            for i, p in enumerate(model_params):
                if p.requires_grad:
                    fast_weights.append(p)  # save original weight
                    fast_order.append(i)
                if not p.requires_grad:
                    frozen_weights.append(p)  # save  weight
                    frozen_order.append(i)

            for step_i in range(args.meta_inner_step):
                weights = frozen_weights + fast_weights
                order = frozen_order + fast_order
                weights = restore_original_order(weights, order)
                result = forward_with_params(model, support_set_x, weights)
                pred_support = result['pred']
                loss_MSE_support = ((pred_support - support_set_y) ** 2).mean()
                if model.auxiliary_task:
                    regression_support = result['regression']
                    support_set_x_norm = norm(support_set_x)
                    target_support = support_set_x_norm.reshape([regression_support.shape[0], support_set_x_norm.shape[-1]]).reshape(regression_support.shape)
                    loss_auxiliary_support = ((regression_support - target_support) ** 2).mean()
                    loss_support = loss_MSE_support + loss_auxiliary_support
                else:
                    loss_support = loss_MSE_support
                Epoch_Labels_support.append(support_set_y.detach().cpu())
                Epoch_preds_support.append(pred_support.detach().cpu())
                grads = torch.autograd.grad(loss_support, fast_weights, retain_graph=False)
                fast_weights = [w - args.meta_inner_lr * g for w, g in zip(fast_weights, grads)]

            #
            with torch.no_grad():
                adapt_weight = frozen_weights + fast_weights
                order = frozen_order + fast_order
                adapt_weight = restore_original_order(adapt_weight, order)
                result = forward_with_params(model, query_set_x, adapt_weight)
                pred_query = result['pred']
                loss_MSE_query = ((pred_query - query_set_y) ** 2).mean()
                if model.auxiliary_task:
                    regression_query = result['regression']
                    query_set_x_norm = norm(query_set_x)
                    target_query = query_set_x_norm.reshape(
                        [regression_query.shape[0], query_set_x_norm.shape[-1]]).reshape(regression_query.shape)
                    loss_auxiliary_query = ((regression_query - target_query) ** 2).mean()
                    loss_query = loss_MSE_query + loss_auxiliary_query
                else:
                    loss_query = loss_MSE_query

                Epoch_Labels_query.append(query_set_y.detach().cpu())
                Epoch_preds_query.append(pred_query.detach().cpu())
                task_losses.append(loss_query.item())

            end = time.time()
            task_time.update(end - start)
            MSE_loss.update(loss_MSE_query.item(), query_set_y.shape[0])
            if model.auxiliary_task:
                auxiliary_loss.update(loss_auxiliary_query.item(), target_query.shape[0])

        progress.display(batch_i)

        meta_loss = sum(task_losses) / len(task_losses)
        print(f'meta loss on query set {meta_loss}')

    Epoch_Labels_support = torch.cat(Epoch_Labels_support, dim=0).squeeze()
    Epoch_preds_support = torch.cat(Epoch_preds_support, dim=0).squeeze()
    Epoch_Labels_query = torch.cat(Epoch_Labels_query, dim=0).squeeze()
    Epoch_preds_query = torch.cat(Epoch_preds_query, dim=0).squeeze()
    Epoch_error_query = abs(Epoch_Labels_query-Epoch_preds_query)
    Label_pred_dict = {'Label':Epoch_Labels_query, 'pred':Epoch_preds_query}
    torch.save(Label_pred_dict, 'Label_pred_data\Label_pred_dict.pth')
    less_5 = sum(Epoch_error_query < 5)/len(Epoch_error_query)
    less_10 = sum(Epoch_error_query < 10) / len(Epoch_error_query)
    less_15 = sum(Epoch_error_query < 15) / len(Epoch_error_query)
    print(f'cumulative error < 5:{less_5}')
    print(f'cumulative error < 10:{less_10}')
    print(f'cumulative error < 15:{less_15}')
    metric_epoch_support = regression_metric(Epoch_preds_support.numpy(), Epoch_Labels_support.numpy())
    metric_epoch_query = regression_metric(Epoch_preds_query.numpy(),
                                             Epoch_Labels_query.numpy())
    log_and_print(metric_epoch_support, epoch, args.save_path, save_name+'_support')
    log_and_print(metric_epoch_query, epoch, args.save_path, save_name+'_query')

    return metric_epoch_query


def transfer_learning_few_shot(data_loader, model, optimizer, epoch, args, save_name='validating'):
    task_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.3f')
    progress = ProgressMeter(
        len(data_loader),
        [task_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    Epoch_preds_query = []
    Epoch_Labels_query = []

    pre_train = model.state_dict()
    for batch_i, (PPG_IMFs_batch, Label_batch) in enumerate(data_loader):
        task_losses = []
        for person_i in range(len(PPG_IMFs_batch)):
            start = time.time()
            PPG_IMFs = PPG_IMFs_batch[person_i]
            Label = Label_batch[person_i]

            support_set_x, support_set_y, query_set_x, query_set_y = data_label_split(PPG_IMFs, Label, support_num=args.k_shot)
            if args.gpu is not None:
                support_set_x = support_set_x.cuda(args.gpu, non_blocking=True).float()
                support_set_y = support_set_y.cuda(args.gpu, non_blocking=True).float()
                query_set_x = query_set_x.cuda(args.gpu, non_blocking=True).float()
                query_set_y = query_set_y.cuda(args.gpu, non_blocking=True).float()

            model.load_state_dict(pre_train)

            # inner loops
            for epoch in range(args.meta_inner_step):
                result = model(support_set_x)
                pred = result['pred']
                loss = ((pred.squeeze() - support_set_y.squeeze()) ** 2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            with torch.no_grad():
                result_query = model(query_set_x)
                pred_query = result_query['pred']
                loss_query = ((pred_query.squeeze() - query_set_y.squeeze())**2).mean()
                Epoch_Labels_query.append(query_set_y.squeeze().detach().cpu())
                Epoch_preds_query.append(pred_query.squeeze().detach().cpu())
                task_losses.append(loss_query.item())

            torch.cuda.empty_cache()

            end = time.time()
            task_time.update(end - start)
            losses.update(loss_query, query_set_x.shape[0])

            progress.display(person_i + batch_i)

        meta_loss = sum(task_losses) / len(task_losses)
        print(f'loss on query set {meta_loss}')

    Epoch_Labels_query = torch.cat(Epoch_Labels_query, dim=0).squeeze()
    Epoch_preds_query = torch.cat(Epoch_preds_query, dim=0).squeeze()
    metric_epoch_query = regression_metric(Epoch_preds_query.numpy(),
                                             Epoch_Labels_query.numpy())
    log_and_print(metric_epoch_query, epoch, args.save_path, save_name)

    return metric_epoch_query


def data_label_split(x, y, support_num=5, query_num=5, shuffle=True):
    if shuffle:
        perm = torch.randperm(len(x))
        support_x = [x[i] for i in perm[:support_num]]
        support_y = [y[i] for i in perm[:support_num]]
        query_x = [x[i] for i in perm[support_num:support_num+query_num]]
        query_y = [y[i] for i in perm[support_num:support_num+query_num]]
        support_x = torch.stack(support_x, dim=0)
        support_y = torch.stack(support_y, dim=0)
        query_x = torch.stack(query_x, dim=0)
        query_y = torch.stack(query_y, dim=0)
    else:
        y_sorted, indices = torch.sort(y)
        x_sorted = x[indices]
        interval = (torch.arange(support_num) * len(x) / support_num).long()
        r = (torch.rand(support_num) * len(x) / support_num).long()
        select_indices = interval + r
        support_x = x_sorted[select_indices]
        support_y = y_sorted[select_indices]
        mask = torch.ones(len(x), dtype=torch.bool)
        mask[select_indices] = False
        x_left = x_sorted[mask]
        y_left = y_sorted[mask]
        perm = torch.randperm(len(x)-support_num)
        query_x = [x_left[i] for i in perm[:query_num]]
        query_y = [y_left[i] for i in perm[:query_num]]
        query_x = torch.stack(query_x, dim=0)
        query_y = torch.stack(query_y, dim=0)
    return support_x, support_y, query_x, query_y


def forward_with_params(model, x, params):
    param_dict = {}
    for i, (name, _) in enumerate(model.named_parameters()):
        param_dict[name] = params[i]
    return torch.func.functional_call(model, param_dict, x, strict=True)


def restore_original_order(shuffled_list, original_indices):
    """
        shuffled_list: 打乱顺序后的列表
        original_indices: 记录原始顺序的索引列表
    """
    # 创建一个与原始列表大小相同的新列表
    restored = [None] * len(shuffled_list)

    # 根据原始索引将元素放回原位
    for new_pos, original_pos in enumerate(original_indices):
        restored[original_pos] = shuffled_list[new_pos]

    return restored

def grads_to_vec(grads):
    return torch.cat([g.view(-1) for g in grads if g is not None])