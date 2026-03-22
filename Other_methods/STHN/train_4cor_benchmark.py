import os
import time
import logging
import math
from datetime import datetime
from uuid import uuid4

import cv2
import numpy as np
import torch
from tqdm import tqdm
import kornia.geometry.transform as tgm

import commons
import parser_benchmark as parser
import datasets_benchmark_json as datasets


def _flow_to_4cor(flow):
    flow_4cor = torch.zeros((flow.shape[0], 2, 2, 2), device=flow.device, dtype=flow.dtype)
    flow_4cor[:, :, 0, 0] = flow[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow[:, :, -1, -1]
    return flow_4cor


def _center_offset_from_4cor(four_disp, resize_width):
    four_point_org_single = torch.zeros((1, 2, 2, 2), dtype=four_disp.dtype, device=four_disp.device)
    four_point_org_single[:, :, 0, 0] = torch.tensor([0.0, 0.0], dtype=four_disp.dtype, device=four_disp.device)
    four_point_org_single[:, :, 0, 1] = torch.tensor([resize_width - 1.0, 0.0], dtype=four_disp.dtype, device=four_disp.device)
    four_point_org_single[:, :, 1, 0] = torch.tensor([0.0, resize_width - 1.0], dtype=four_disp.dtype, device=four_disp.device)
    four_point_org_single[:, :, 1, 1] = torch.tensor([resize_width - 1.0, resize_width - 1.0], dtype=four_disp.dtype, device=four_disp.device)

    four_point_1 = four_disp + four_point_org_single
    four_point_org = four_point_org_single.repeat(four_point_1.shape[0], 1, 1, 1).flatten(2).permute(0, 2, 1).contiguous()
    four_point_1 = four_point_1.flatten(2).permute(0, 2, 1).contiguous()
    H = tgm.get_perspective_transform(four_point_org, four_point_1)

    center_T = torch.tensor(
        [resize_width / 2.0 - 0.5, resize_width / 2.0 - 0.5, 1.0],
        dtype=four_disp.dtype,
        device=four_disp.device,
    ).unsqueeze(1).unsqueeze(0).repeat(H.shape[0], 1, 1)
    w = torch.bmm(H, center_T).squeeze(2)
    center_offset = w[:, :2] / w[:, 2].unsqueeze(1) - center_T[:, :2].squeeze(2)
    return center_offset


def _evaluate_split(model, args, split='val', max_batches=-1):
    val_loader = datasets.fetch_dataloader(args, split=split)
    model.netG.eval()
    if args.two_stages and hasattr(model, 'netG_fine'):
        model.netG_fine.eval()

    error_m_list = []
    valid_count = 0
    skipped_count = 0
    latency_ms_list = []
    frame_counter = 0
    use_cuda_timing = torch.cuda.is_available() and str(args.device).startswith('cuda')
    meter_per_resize_px = float(args.crop_size_m) / float(args.resize_width)

    with torch.no_grad():
        for batch_idx, data_blob in enumerate(tqdm(val_loader, desc=f'eval-{split}', leave=False)):
            if max_batches > 0 and batch_idx >= max_batches:
                break

            if len(data_blob) >= 9:
                image1, image2, flow_gt, _, _, _, _, _, valid_gt = [x for x in data_blob]
                valid_mask = (valid_gt > 0.5)
                if torch.is_tensor(valid_mask):
                    valid_mask = valid_mask.bool()
                if valid_mask.sum().item() == 0:
                    skipped_count += int(valid_gt.shape[0])
                    continue
                image1 = image1[valid_mask]
                image2 = image2[valid_mask]
                flow_gt = flow_gt[valid_mask]
                skipped_count += int(valid_gt.shape[0] - valid_mask.sum().item())

            if split == 'test':
                pred_list = []
                gt_list = []
                for i in range(image1.shape[0]):
                    image1_i = image1[i:i + 1]
                    image2_i = image2[i:i + 1]
                    flow_gt_i = flow_gt[i:i + 1]

                    if use_cuda_timing:
                        torch.cuda.synchronize()
                    tic = time.perf_counter()
                    model.set_input(image1_i, image2_i, flow_gt_i)
                    model.forward()
                    if use_cuda_timing:
                        torch.cuda.synchronize()
                    latency_ms = (time.perf_counter() - tic) * 1000.0

                    frame_counter += 1
                    latency_ms_list.append(latency_ms)
                    # logging.info(f"[test-latency] frame={frame_counter} time_ms={latency_ms:.3f}")

                    pred_list.append(model.four_pred.detach())
                    gt_list.append(flow_gt_i.to(model.four_pred.device))

                four_pred = torch.cat(pred_list, dim=0)
                flow_4cor_gt = _flow_to_4cor(torch.cat(gt_list, dim=0))
            else:
                model.set_input(image1, image2, flow_gt)
                model.forward()
                four_pred = model.four_pred.detach()
                flow_4cor_gt = _flow_to_4cor(flow_gt.to(four_pred.device))

            pred_center_offset = _center_offset_from_4cor(four_pred, args.resize_width)
            gt_center_offset = _center_offset_from_4cor(flow_4cor_gt, args.resize_width)

            error_px = torch.norm(pred_center_offset - gt_center_offset, dim=1)
            error_m = error_px * meter_per_resize_px
            error_m_list.append(error_m.detach().cpu().numpy())
            valid_count += int(error_m.shape[0])

    if len(error_m_list) == 0:
        metrics = {
            'count': 0,
            'skipped': skipped_count,
            'loc_err_mean_m': float('nan'),
            'loc_err_std_m': float('nan'),
            'recall_5m': float('nan'),
            'recall_10m': float('nan'),
            'recall_20m': float('nan'),
            'precision_5m': float('nan'),
            'precision_10m': float('nan'),
            'precision_20m': float('nan'),
        }
    else:
        errors = np.concatenate(error_m_list, axis=0)
        r5 = float(np.mean(errors < 5.0))
        r10 = float(np.mean(errors < 10.0))
        r20 = float(np.mean(errors < 20.0))
        metrics = {
            'count': int(errors.shape[0]),
            'skipped': skipped_count,
            'loc_err_mean_m': float(np.mean(errors)),
            'loc_err_std_m': float(np.std(errors)),
            'recall_5m': r5,
            'recall_10m': r10,
            'recall_20m': r20,
            'precision_5m': r5,
            'precision_10m': r10,
            'precision_20m': r20,
        }

    if split == 'test' and len(latency_ms_list) > 0:
        latency_arr = np.asarray(latency_ms_list, dtype=np.float64)
        metrics['latency_mean_ms'] = float(np.mean(latency_arr))
        metrics['latency_median_ms'] = float(np.median(latency_arr))
        metrics['latency_p95_ms'] = float(np.percentile(latency_arr, 95))
        metrics['latency_min_ms'] = float(np.min(latency_arr))
        metrics['latency_max_ms'] = float(np.max(latency_arr))
        logging.info(
            f"[test-latency-summary] frames={len(latency_arr)} "
            f"mean={metrics['latency_mean_ms']:.3f}ms "
            f"median={metrics['latency_median_ms']:.3f}ms "
            f"p95={metrics['latency_p95_ms']:.3f}ms "
            f"min={metrics['latency_min_ms']:.3f}ms "
            f"max={metrics['latency_max_ms']:.3f}ms"
        )

    model.netG.train()
    if args.two_stages and hasattr(model, 'netG_fine'):
        model.netG_fine.train()
    return metrics


def _safe_load_checkpoint(model, ckpt_path, args):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'netG' in checkpoint:
        model.netG.load_state_dict(checkpoint['netG'], strict=False)
    else:
        model.netG.load_state_dict(checkpoint, strict=False)

    if args.two_stages and hasattr(model, 'netG_fine') and checkpoint.get('netG_fine') is not None:
        model.netG_fine.load_state_dict(checkpoint['netG_fine'], strict=False)


def _save_checkpoint(model, save_dir, step):
    checkpoint = {
        'netG': model.netG.state_dict(),
        'netG_fine': model.netG_fine.state_dict() if hasattr(model, 'netG_fine') else None,
        'step': step,
    }
    torch.save(checkpoint, os.path.join(save_dir, f'step_{step:07d}.pth'))
    torch.save(checkpoint, os.path.join(save_dir, 'latest.pth'))


def _build_unique_save_dir(save_root, run_name, max_retry=20):
    for _ in range(max_retry):
        run_tag = f"{run_name}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}-{uuid4()}"
        candidate = os.path.join(save_root, run_tag)
        if not os.path.exists(candidate):
            return candidate, run_tag
    raise RuntimeError('Failed to allocate a unique logging directory.')


def _tensor_to_bgr_uint8(img_tensor):
    arr = img_tensor.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    arr = np.clip(arr, 0.0, 1.0)
    arr = (arr * 255.0).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _render_batch_visualization(model, sat_batch, query_batch, args, save_dir, step):
    vis_dir = os.path.join(save_dir, 'train_vis')
    os.makedirs(vis_dir, exist_ok=True)

    num_items = min(len(sat_batch), int(args.vis_max_items))
    if num_items <= 0:
        return

    sat_size_px = float(getattr(args, 'sat_size_px', max(2, int(round(float(args.crop_size_m))))))
    alpha = sat_size_px / float(args.resize_width)
    org = model.four_point_org_single[0].detach().cpu().numpy()
    pred_4cor = model.four_pred.detach().cpu().numpy()
    gt_4cor = model.flow_4cor.detach().cpu().numpy()

    tiles = []
    for i in range(num_items):
        sat_img = _tensor_to_bgr_uint8(sat_batch[i])
        qry_img = _tensor_to_bgr_uint8(query_batch[i])

        sat_h, sat_w = sat_img.shape[:2]
        qry_img = cv2.resize(qry_img, (sat_w, sat_h), interpolation=cv2.INTER_LINEAR)

        gt_corners = (org + gt_4cor[i]).reshape(2, -1)
        pred_corners = (org + pred_4cor[i]).reshape(2, -1)

        gt_x = int(np.mean(gt_corners[0]) * alpha)
        gt_y = int(np.mean(gt_corners[1]) * alpha)
        pred_x = int(np.mean(pred_corners[0]) * alpha)
        pred_y = int(np.mean(pred_corners[1]) * alpha)

        gt_x = max(0, min(sat_w - 1, gt_x))
        gt_y = max(0, min(sat_h - 1, gt_y))
        pred_x = max(0, min(sat_w - 1, pred_x))
        pred_y = max(0, min(sat_h - 1, pred_y))

        cv2.circle(sat_img, (gt_x, gt_y), 8, (0, 255, 0), -1, lineType=cv2.LINE_AA)
        cv2.circle(sat_img, (pred_x, pred_y), 8, (0, 0, 255), -1, lineType=cv2.LINE_AA)
        cv2.line(sat_img, (gt_x, gt_y), (pred_x, pred_y), (0, 255, 255), 2, lineType=cv2.LINE_AA)
        cv2.putText(sat_img, 'GT', (gt_x + 10, gt_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(sat_img, 'Pred', (pred_x + 10, pred_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

        pair = np.concatenate([qry_img, sat_img], axis=1)
        cv2.putText(pair, f'idx={i}', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(pair)

    ncols = int(math.ceil(math.sqrt(len(tiles))))
    nrows = int(math.ceil(len(tiles) / ncols))
    th, tw = tiles[0].shape[:2]
    canvas = np.zeros((nrows * th, ncols * tw, 3), dtype=np.uint8)

    for idx, tile in enumerate(tiles):
        r = idx // ncols
        c = idx % ncols
        canvas[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = tile

    out_path = os.path.join(vis_dir, f'step_{step:07d}.jpg')
    cv2.imwrite(out_path, canvas)
    return out_path


def main(args):
    from model.network import STHN
    from utils import count_parameters, setup_seed

    save_dir, run_tag = _build_unique_save_dir(args.save_root, args.name)
    commons.setup_logging(save_dir, console='info')
    setup_seed(args.seed)

    logging.info('Creating model...')
    model = STHN(args, for_training=True)
    model.setup()
    model.netG.train()
    if args.two_stages and hasattr(model, 'netG_fine'):
        model.netG_fine.train()
    logging.info(f'Parameter Count (netG): {count_parameters(model.netG)}')

    if args.restore_ckpt:
        logging.info(f'Loading checkpoint from: {args.restore_ckpt}')
        _safe_load_checkpoint(model, args.restore_ckpt, args)

    train_loader = datasets.fetch_dataloader(args, split='train')
    logging.info(f'Train samples: {len(train_loader.dataset)}')

    swanlab_run = None
    if args.use_swanlab:
        try:
            import swanlab
            swanlab_run = swanlab.init(
                project=args.swanlab_project,
                experiment_name=args.swanlab_experiment or run_tag,
                config=vars(args),
            )
        except Exception as exc:
            logging.warning(f'SwanLab init failed, fallback to no-op: {exc}')
            swanlab_run = None

    total_steps = 0
    skipped_invalid = 0
    last_valid_ratio = 1.0
    last_gt_disp_mean = 0.0
    last_gt_disp_std = 0.0
    current_offset_range_m = float(args.offset_range_m)
    while total_steps < args.num_steps:
        for data_blob in tqdm(train_loader, desc='finetune', leave=False):
            tic = time.time()
            if args.curriculum_offset_enable:
                warmup_steps = max(1, int(args.curriculum_offset_warmup_steps))
                progress = min(float(total_steps) / float(warmup_steps), 1.0)
                current_offset_range_m = float(args.curriculum_offset_start) + (
                    float(args.curriculum_offset_end) - float(args.curriculum_offset_start)
                ) * progress
                args.offset_range_m = current_offset_range_m
            else:
                current_offset_range_m = float(args.offset_range_m)

            if len(data_blob) >= 9:
                image1, image2, flow, _, query_utm, database_utm, _, _, valid_gt = [x for x in data_blob]
                valid_mask = (valid_gt > 0.5)
                if torch.is_tensor(valid_mask):
                    valid_mask = valid_mask.bool()
                batch_size_now = int(valid_gt.shape[0]) if torch.is_tensor(valid_gt) else args.batch_size
                valid_count = int(valid_mask.sum().item()) if torch.is_tensor(valid_mask) else 0
                last_valid_ratio = float(valid_count) / float(max(1, batch_size_now))

                if valid_count > 0:
                    gt_offset = (query_utm - database_utm)[valid_mask]
                    gt_disp = torch.sqrt(gt_offset[:, 0, 0] ** 2 + gt_offset[:, 0, 1] ** 2)
                    last_gt_disp_mean = float(gt_disp.mean().item())
                    last_gt_disp_std = float(gt_disp.std().item()) if gt_disp.numel() > 1 else 0.0
                if valid_count == 0:
                    skipped_invalid += batch_size_now
                    continue

                image1 = image1[valid_mask]
                image2 = image2[valid_mask]
                flow = flow[valid_mask]
                query_utm = query_utm[valid_mask]
                database_utm = database_utm[valid_mask]
                skipped_invalid += int(valid_gt.shape[0]) - valid_count
            else:
                image1, image2, flow, _, query_utm, database_utm, _, _ = [x for x in data_blob]
            model.set_input(image1, image2, flow)
            metrics = model.optimize_parameters()
            model.update_learning_rate()

            total_steps += 1
            elapsed = time.time() - tic
            lr_value = model.scheduler_G.get_last_lr()[0]
            normalized_ce = float(metrics['ce_loss']) / max(1e-6, float(last_gt_disp_mean))
            normalized_mace = float(metrics['mace']) / max(1e-6, float(last_gt_disp_mean))

            if total_steps % args.print_freq == 0:
                logging.info(
                    f"step={total_steps} mace={metrics['mace']:.4f} "
                    f"ce={metrics['ce_loss']:.4f} loss={metrics['G_loss']:.4f} "
                    f"norm_ce={normalized_ce:.4f} norm_mace={normalized_mace:.4f} "
                    f"lr={lr_value:.6e} time={elapsed:.3f}s "
                    f"offset_range_m={current_offset_range_m:.2f} "
                    f"skipped_invalid={skipped_invalid} valid_ratio={last_valid_ratio:.3f} "
                    f"gt_disp_mean={last_gt_disp_mean:.2f} gt_disp_std={last_gt_disp_std:.2f}"
                )

            if swanlab_run is not None:
                try:
                    import swanlab
                    swanlab.log(
                        {
                            'step': total_steps,
                            'mace': metrics['mace'],
                            'ce_loss': metrics['ce_loss'],
                            'G_loss': metrics['G_loss'],
                            'norm_ce': normalized_ce,
                            'norm_mace': normalized_mace,
                            'lr': lr_value,
                            'time': elapsed,
                            'offset_range_m': current_offset_range_m,
                            'skipped_invalid': skipped_invalid,
                            'valid_ratio': last_valid_ratio,
                            'gt_disp_mean': last_gt_disp_mean,
                            'gt_disp_std': last_gt_disp_std,
                        }
                    )
                except Exception as exc:
                    logging.warning(f'SwanLab log failed at step={total_steps}: {exc}')

            if args.enable_train_vis and total_steps % args.vis_interval == 0:
                vis_path = _render_batch_visualization(
                    model=model,
                    sat_batch=image1,
                    query_batch=image2,
                    args=args,
                    save_dir=save_dir,
                    step=total_steps,
                )
                if vis_path is not None:
                    logging.info(f'saved batch visualization: {vis_path}')
                    if swanlab_run is not None:
                        try:
                            import swanlab
                            if hasattr(swanlab, 'Image'):
                                swanlab.log({'train_vis': swanlab.Image(vis_path), 'step': total_steps})
                            else:
                                swanlab.log({'train_vis_step': total_steps})
                        except Exception as exc:
                            logging.warning(f'SwanLab image log failed at step={total_steps}: {exc}')

            if total_steps % args.save_freq == 0:
                _save_checkpoint(model, save_dir, total_steps)

            if args.run_val and args.eval_val_every > 0 and total_steps % args.eval_val_every == 0:
                val_metrics = _evaluate_split(model, args, split='val', max_batches=args.eval_max_batches)
                logging.info(
                    f"[val] step={total_steps} n={val_metrics['count']} skipped={val_metrics['skipped']} "
                    f"mean={val_metrics['loc_err_mean_m']:.3f}m std={val_metrics['loc_err_std_m']:.3f}m "
                    f"R@5m={val_metrics['recall_5m']:.3f} R@10m={val_metrics['recall_10m']:.3f} R@20m={val_metrics['recall_20m']:.3f}"
                )
                if swanlab_run is not None:
                    try:
                        import swanlab
                        swanlab.log({
                            'val_step': total_steps,
                            'val_loc_err_mean_m': val_metrics['loc_err_mean_m'],
                            'val_loc_err_std_m': val_metrics['loc_err_std_m'],
                            'val_recall_5m': val_metrics['recall_5m'],
                            'val_recall_10m': val_metrics['recall_10m'],
                            'val_recall_20m': val_metrics['recall_20m'],
                        })
                    except Exception as exc:
                        logging.warning(f'SwanLab val log failed at step={total_steps}: {exc}')

            if total_steps >= args.num_steps:
                break

    _save_checkpoint(model, save_dir, total_steps)
    logging.info(f'Finetune finished, final step={total_steps}')
    logging.info(f'Checkpoints saved in: {save_dir}')

    if args.run_test:
        test_metrics = _evaluate_split(model, args, split='test', max_batches=args.eval_max_batches)
        logging.info(
            f"[test] n={test_metrics['count']} skipped={test_metrics['skipped']} "
            f"mean={test_metrics['loc_err_mean_m']:.3f}m std={test_metrics['loc_err_std_m']:.3f}m "
            f"R@5m={test_metrics['recall_5m']:.3f} R@10m={test_metrics['recall_10m']:.3f} R@20m={test_metrics['recall_20m']:.3f} "
            f"P@5m={test_metrics['precision_5m']:.3f} P@10m={test_metrics['precision_10m']:.3f} P@20m={test_metrics['precision_20m']:.3f}"
        )
        if swanlab_run is not None:
            try:
                import swanlab
                swanlab.log({
                    'test_loc_err_mean_m': test_metrics['loc_err_mean_m'],
                    'test_loc_err_std_m': test_metrics['loc_err_std_m'],
                    'test_recall_5m': test_metrics['recall_5m'],
                    'test_recall_10m': test_metrics['recall_10m'],
                    'test_recall_20m': test_metrics['recall_20m'],
                    'test_precision_5m': test_metrics['precision_5m'],
                    'test_precision_10m': test_metrics['precision_10m'],
                    'test_precision_20m': test_metrics['precision_20m'],
                })
            except Exception as exc:
                logging.warning(f'SwanLab test log failed: {exc}')

    if swanlab_run is not None:
        try:
            import swanlab
            swanlab.finish()
        except Exception:
            pass


if __name__ == '__main__':
    args = parser.parse_arguments()
    main(args)
