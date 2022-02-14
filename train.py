import open3d as o3d
import time, os, sys, glob, argparse
import numpy as np
import torch
import MinkowskiEngine as ME
import multiprocessing as mp
from model.Network import MyNet

from data import make_data_loader
from utils.loss import get_metrics
from utils.pc_error_wrapper import pc_error

from tensorboardX import SummaryWriter
import logging


def getlogger(logdir):
  logger = logging.getLogger(__name__)
  logger.setLevel(level = logging.INFO)
  
  handler = logging.FileHandler(os.path.join(logdir, 'log.txt'))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%m/%d %H:%M:%S')
  handler.setFormatter(formatter)
  
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  console.setFormatter(formatter)
  
  logger.addHandler(handler)
  logger.addHandler(console)
  
  return logger


def parse_args():
  parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument("--dataset", default='../Data/ShapeNet/')
  parser.add_argument("--downsample", default=8, help='Downsample Rate')
  parser.add_argument("--num_test", type=int, default=1024, help='how many of the dataset use for testing')
  parser.add_argument("--dataset_8i", default='../Data/8i_test/8x/')
  parser.add_argument("--dataset_8i_GT", default='../Data/8i_test/orig/')
  parser.add_argument("--last_kernel_size", type=int, default=7, help='The final layer kernel size, coordinates get expanded by this')
  
  parser.add_argument("--init_ckpt", default='ckpts/8x_0x_ks7/iter2000.pth')
  parser.add_argument("--reset", default=False, action='store_true', help='reset training')
  
  parser.add_argument("--lr", type=float, default=8e-4)
  parser.add_argument("--batch_size", type=int, default=4)
  parser.add_argument("--global_step", type=int, default=int(64000))
  parser.add_argument("--base_step", type=int, default=int(100),  help='frequency for recording state.')
  parser.add_argument("--test_step", type=int, default=int(2000),  help='frequency for test and save.')
  # parser.add_argument("--random_seed", type=int, default=4, help='random_seed.')
  
  parser.add_argument("--max_norm", type=float, default=1.,  help='max norm for gradient clip, close if 0')
  parser.add_argument("--clip_value", type=float, default=0,  help='max value for gradient clip, close if 0')
  
  parser.add_argument("--logdir", type=str, default='logs', help="logger direction.")
  parser.add_argument("--ckptdir", type=str, default='ckpts', help="ckpts direction.")
  parser.add_argument("--prefix", type=str, default='8x_0x_ks7', help="prefix of checkpoints/logger, etc.")
  parser.add_argument("--lr_gamma", type=float, default=0.5, help="gamma for lr_scheduler.")
  parser.add_argument("--lr_step", type=int, default=6000, help="step for adjusting lr_scheduler.")  

  args = parser.parse_args()
  return args



def kdtree_partition2(pc, pc2, max_num):
    parts = []
    parts2 = []
    
    class KD_node:  
        def __init__(self, point=None, LL = None, RR = None):  
            self.point = point  
            self.left = LL  
            self.right = RR
            
    def createKDTree(root, data, data2):
        if len(data) <= max_num:
            parts.append(data)
            parts2.append(data2)
            return
		
        variances = (np.var(data[:, 0]), np.var(data[:, 1]), np.var(data[:, 2]))
        dim_index = variances.index(max(variances))
        data_sorted = data[np.lexsort(data.T[dim_index, None])]
        data2_sorted = data2[np.lexsort(data2.T[dim_index, None])]
		
        point = data_sorted[int(len(data)/2)]  
  		
        root = KD_node(point)  
        root.left = createKDTree(root.left, data_sorted[: int((len(data) / 2))], data2_sorted[: int((len(data2) / 2))])  
        root.right = createKDTree(root.right, data_sorted[int((len(data) / 2)):], data2_sorted[int((len(data2) / 2)):]) 
        
        return root
    
    init_root = KD_node(None)
    _ = createKDTree(init_root, pc, pc2)
	
    return parts, parts2



### This testing function is made for larger point clouds and uses kd_tree partition.
def test2(model, test_dataloader, logger, writer, writername, step, test_pc_error, args, device, First_eval):

  start_time = time.time()

  # data.
  test_iter = iter(test_dataloader)

  # loss & metrics.
  sum_loss = 0.
  all_metrics = np.zeros((1,3))
  all_pc_errors = np.zeros(3)
  all_pc_errors2 = np.zeros(2)

  # model & crit.
  model.to(device)# to cpu.
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # loop per batch.
  for i in range(len(test_iter)):
    coords, _, coords_T = test_iter.next()
    parts_pc, parts_pc2 = kdtree_partition2(coords[:,1:].numpy(), coords_T[:,1:].numpy(), 70000)
    out_l = []
    out_cls_l = []
    target_l = []
    keep_l = []
    
    # Forward.
    for j,pc in enumerate(parts_pc):
        p = ME.utils.batched_coordinates([pc])
        p2 = ME.utils.batched_coordinates([parts_pc2[j]])
        f = torch.from_numpy(np.vstack(np.expand_dims(np.ones(p.shape[0]), 1))).float()
        x1 = ME.SparseTensor(feats=f, coords=p).to(device)
        
        with torch.no_grad():
            out, out_cls, target, keep = model(x1, coords_T=p2, device=device, prune=True)
        
        out_l.append(out.C[:,1:])
        out_cls_l.append(out_cls.F)
        target_l.append(target)
        keep_l.append(keep)
        
    rec_pc = torch.cat(out_l, 0)
    rec_pc_cls = torch.cat(out_cls_l, 0)
    rec_target = torch.cat(target_l, 0)
    rec_keep = torch.cat(keep_l, 0)
    
    loss = crit(rec_pc_cls.squeeze(), rec_target.type(out_cls.F.dtype).to(device))
    metrics = get_metrics(rec_keep, rec_target)
    
    # get pc_error.
    if test_pc_error:
      GT_pcd = o3d.geometry.PointCloud()
      GT_pcd.points = o3d.utility.Vector3dVector(coords_T[:,1:])
      # ori_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=20, max_nn=20))
      
      GTfile = 'tmp/'+args.prefix+'GT.ply'
      o3d.io.write_point_cloud(GTfile, GT_pcd, write_ascii=True)
      
      rec_pcd = o3d.geometry.PointCloud()
      rec_pcd.points = o3d.utility.Vector3dVector(rec_pc)
      
      recfile = 'tmp/'+args.prefix+'rec.ply'
      o3d.io.write_point_cloud(recfile, rec_pcd, write_ascii=True)
      
      pc_error_metrics = pc_error(infile1=GTfile, infile2=recfile, res=1024)
      pc_errors = [pc_error_metrics['mse1,PSNR (p2point)'][0], 
                  pc_error_metrics['mse2,PSNR (p2point)'][0], 
                  pc_error_metrics['mseF,PSNR (p2point)'][0]]
      
      if First_eval:
          in_pcd = o3d.geometry.PointCloud()
          in_pcd.points = o3d.utility.Vector3dVector(coords[:,1:].numpy())
          
          infile = 'tmp/'+args.prefix+'in.ply'
          o3d.io.write_point_cloud(infile, in_pcd, write_ascii=True)
          
          pc_error_metrics = pc_error(infile1=GTfile, infile2=infile, res=1024)
          pc_errors2 = [pc_error_metrics['mse1,PSNR (p2point)'][0],
                  pc_error_metrics['mseF,PSNR (p2point)'][0]]
          
    # record.
    with torch.no_grad():
      sum_loss += loss.item()
      all_metrics += np.array(metrics)
      if test_pc_error:
        all_pc_errors += np.array(pc_errors)
        if First_eval:
            all_pc_errors2 += np.array(pc_errors2)

  print('======testing time:', round(time.time() - start_time, 4), 's')

  sum_loss /= len(test_iter)
  all_metrics /= len(test_iter)
  if test_pc_error:
    all_pc_errors /= len(test_iter)
    if First_eval:
        all_pc_errors2 /= len(test_iter)

  # logger.
  logger.info(f'\nIteration: {step}')
  logger.info(f'Sum Loss: {sum_loss:.4f}')
  logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')
  if test_pc_error:
    logger.info(f'all_pc_errors: {np.round(all_pc_errors, 4).tolist()}')
  
  # writer.
  writer.add_scalars(main_tag=writername+'/losses', 
                    tag_scalar_dict={'sum_loss': sum_loss}, 
                    global_step=step)
  
  writer.add_scalars(main_tag=writername+'/metrics', 
                    tag_scalar_dict={'Precision': all_metrics[0,0],
                                     'Recall': all_metrics[0,1],
                                     'IoU': all_metrics[0,2]},
                    global_step=step)
  
  if test_pc_error:
    writer.add_scalars(main_tag=writername+'/out_pc_errors', 
                  tag_scalar_dict={'p2point1': all_pc_errors[0],
                                    'p2point2': all_pc_errors[1],
                                    'p2pointF': all_pc_errors[2],},
                  global_step=step)
    if First_eval:
        writer.add_scalars(main_tag=writername+'/In_pc_errors', 
                      tag_scalar_dict={'p2point1': all_pc_errors2[0],
                                        'p2pointF': all_pc_errors2[1]},
                      global_step=step)
  return


### This testing function is made for smaller point clouds and does not use kd_tree partition.
def test1(model, test_dataloader, logger, writer, writername, step, args, device):

  start_time = time.time()

  # data.
  test_iter = iter(test_dataloader)

  # loss & metrics.
  sum_loss = 0.
  all_metrics = np.zeros((1,3))

  # model & crit.
  model.to(device)# to cpu.
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # loop per batch.
  for i in range(len(test_iter)):
    coords, feats, coords_T = test_iter.next()
    x = ME.SparseTensor(feats=feats, coords=coords).to(device)
    
    # Forward.
    with torch.no_grad():
        out, out_cls, target, keep = model(x, coords_T=coords_T, device=device, prune=True)
    
    loss = crit(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
    metrics = get_metrics(keep, target)
    
    # record.
    with torch.no_grad():
      sum_loss += loss.item()
      all_metrics += np.array(metrics)

  print('======testing time:', round(time.time() - start_time, 4), 's')

  sum_loss /= len(test_iter)
  all_metrics /= len(test_iter)

  # logger.
  logger.info(f'\nIteration: {step}')
  logger.info(f'Sum Loss: {sum_loss:.4f}')
  logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')

  # writer.
  writer.add_scalars(main_tag=writername+'/losses', 
                    tag_scalar_dict={'sum_loss': sum_loss}, 
                    global_step=step)

  writer.add_scalars(main_tag=writername+'/metrics', 
                    tag_scalar_dict={'Precision': all_metrics[0,0],
                                     'Recall': all_metrics[0,1],
                                     'IoU': all_metrics[0,2]},
                    global_step=step)    
  return


def train(model, train_dataloader, test_dataloader, test_dataloader2, logger, writer, args, device):
  # Optimizer.
  optimizer = torch.optim.Adam([{"params":model.parameters(), 'lr':args.lr}], 
                                betas=(0.9, 0.999), weight_decay=1e-4)
  # adjust lr.
  scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
  # criterion.
  crit = torch.nn.BCEWithLogitsLoss()

  # define checkpoints direction.
  ckptdir = os.path.join(args.ckptdir, args.prefix)
  if not os.path.exists(ckptdir):
    logger.info(f'Make direction for saving checkpoints: {ckptdir}')
    os.makedirs(ckptdir)

  # Load checkpoints.
  start_step = 1
  if args.init_ckpt == '':
    logger.info('Random initialization.')
    First_eval = True
  else:
    # load params from checkpoints.
    logger.info(f'Load checkpoint from {args.init_ckpt}')
    ckpt = torch.load(args.init_ckpt)
    model.load_state_dict(ckpt['model'])
    First_eval = False
    # load start step & optimizer.
    if not args.reset:
        try:
            optimizer.load_state_dict(ckpt['optimizer'])
        except:
            print("Optimizer State Load Failed")    
        start_step = ckpt['step'] + 1

  # start step.
  logger.info(f'LR: {scheduler.get_lr()}')
  print('==============', start_step)

  train_iter = iter(train_dataloader)
  start_time = time.time()
  sum_loss = 0
  all_metrics = np.zeros((1,3))
  

  for i in range(start_step, args.global_step+1):
    if i%10==0:
        print(i)
    optimizer.zero_grad()
    
    s = time.time()
    coords, feats, coords_T = train_iter.next()
    dataloader_time = time.time() - s

    x = ME.SparseTensor(feats=feats, coords=coords).to(device)

    if x.__len__() >= 1e10:
      logger.info(f'\n\n\n======= larger than 1e10 ======: {x.__len__()}\n\n\n')
      continue
  
    # Forward.
    _, out_cls, target, keep = model(x, coords_T=coords_T, device=device, prune=False)
    
    loss = crit(out_cls.F.squeeze(), target.type(out_cls.F.dtype).to(device))
    metrics = get_metrics(keep, target)
        
    if torch.isnan(loss) or torch.isinf(loss):
        logger.info(f'\n== loss is nan ==, Step: {i}\n')
        continue
        
    # Backward.    
    loss.backward()
    
    # Optional clip gradient. 
    if args.max_norm != 0:
      # clip by norm
      max_grad_before = max(p.grad.data.abs().max() for p in model.parameters())
      total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
      
      if total_norm > args.max_norm:

        def get_total_norm(parameters, norm_type=2):
          total_norm = 0.
          for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
          total_norm = total_norm ** (1. / norm_type)
          return total_norm

        print('total_norm:',
          '\nBefore: total_norm:,', total_norm, 
          'max grad:', max_grad_before, 
          '\nthreshold:', args.max_norm, 
          '\nAfter:', get_total_norm(model.parameters()), 
          'max grad:', max(p.grad.data.abs().max() for p in model.parameters()))

    if args.clip_value != 0:
      torch.nn.utils.clip_grad_value_(model.parameters(), args.clip_value)
      print('after gradient clip',  max(p.grad.data.abs().max() for p in model.parameters()))
    
    optimizer.step()
    
    # record.
    with torch.no_grad():
      sum_loss += loss.item()
      all_metrics += np.array(metrics)
    
    # Display.
    if i % args.base_step == 0:
      # average.
      with torch.no_grad():
        sum_loss /= args.base_step
        all_metrics /= args.base_step

      if np.isinf(sum_loss):
        logger.info('inf error!')
        sys.exit(0)
        
      if np.isnan(sum_loss):
        logger.info('NaN error!')
        sys.exit(0)

      # logger.
      logger.info(f'\nIteration: {i}')
      logger.info(f'Running time: {((time.time()-start_time)/60):.2f} min')
      logger.info(f'Data Loading time: {dataloader_time:.5f} s')
      logger.info(f'Sum Loss: {sum_loss:.4f}')
      logger.info(f'Metrics (prec, recal, IOU): {np.round(all_metrics, 4).tolist()}')


      # writer.
      writer.add_scalars(main_tag='train/losses', 
                        tag_scalar_dict={'sum_loss' :sum_loss},
                        global_step=i)

      writer.add_scalars(main_tag='train/metrics', 
                        tag_scalar_dict={'Precision': all_metrics[0,0],
                                         'Recall': all_metrics[0,1],
                                         'IoU': all_metrics[0,2]},
                        global_step=i)
      
      # return 0.
      sum_loss = 0.
      all_metrics = np.zeros((1,3))

      # empty cache.
      torch.cuda.empty_cache()
      
    if i % (args.test_step) == 0:
      logger.info(f'\n==========Evaluation: iter {i}==========')
      # save.
      logger.info(f'save checkpoints: {ckptdir}/iter{str(i)}')
      torch.save({'step': i, 'model': model.state_dict(),
                  'optimizer':  optimizer.state_dict(),
                  }, os.path.join(ckptdir, 'iter' + str(i) + '.pth'))

      # Evaluation.
      logger.info(f'\n=====Evaluation: iter {i} =====')
      with torch.no_grad():
        test1(model=model, test_dataloader=test_dataloader, 
          logger=logger, writer=writer, writername='eval', step=i, args=args, device=device)

      torch.cuda.empty_cache()

    if i % (args.test_step) == 0:
      # Evaluation 8i.
      logger.info(f'\n=====Evaluation: iter {i} 8i =====')
      with torch.no_grad():
        test2(model=model, test_dataloader=test_dataloader2, 
          logger=logger, writer=writer, writername='eval_8i', step=i, test_pc_error=True, args=args, device=device, First_eval=First_eval)
        First_eval=False
      torch.cuda.empty_cache()

      model.to(device)

    if i % int(args.lr_step) == 0:
      scheduler.step()
      logger.info(f'LR: {scheduler.get_lr()}')
    
  writer.close()
    
if __name__ == '__main__':
  args = parse_args()

  logdir = os.path.join(args.logdir, args.prefix)
  if not os.path.exists(logdir):
    os.makedirs(logdir)
  if not os.path.exists('tmp'):
    os.makedirs('tmp')
  logger = getlogger(logdir)
  logger.info(args)
  writer = SummaryWriter(log_dir=logdir)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  logger.info(f'Device:{device}')

  # Load data.
  filedirs = glob.glob(args.dataset+'*.h5')
  filedirs = sorted(filedirs)
  logger.info(f'Files length: {len(filedirs)}')

  train_dataloader = make_data_loader(files=filedirs[int(args.num_test):],
                                      GT_folder=None,
                                      batch_size=args.batch_size,
                                      downsample=args.downsample,
                                      shuffle=True, 
                                      num_workers=mp.cpu_count(),
                                      repeat=True)
  
  test_dataloader = make_data_loader( files=filedirs[:int(args.num_test)],
                                      GT_folder=None,
                                      batch_size=args.batch_size, 
                                      downsample=args.downsample,
                                      shuffle=False, 
                                      num_workers=mp.cpu_count(),
                                      repeat=False)

  # 8i dataset
  eighti_filedirs = glob.glob(args.dataset_8i+'*.ply')
  eighti_filedirs = sorted(eighti_filedirs)
  logger.info(f'Files length: {len(eighti_filedirs)}')

  eighti_dataloader = make_data_loader( files=eighti_filedirs, 
                                        GT_folder=args.dataset_8i_GT,
                                        batch_size=1, 
                                        downsample=args.downsample,
                                        shuffle=False,
                                        num_workers=1,
                                        repeat=False)

  # Network.
  model = MyNet(last_kernel_size=args.last_kernel_size).to(device)
  logger.info(model)

  train(model, train_dataloader, test_dataloader, eighti_dataloader, logger, writer, args, device)

