import os
import argparse
import torch
import random

import csv
import datetime
from torch.nn import functional as F
import numpy as np

from src.models.video_model import DMC
from src.models.image_model import IntraNoAR
from torch.utils.data import DataLoader
from dataload.dataset import *
from tqdm import tqdm
from utils.tools import *
from dataload.dataset_helper import *

metric_list = ['mse', 'ms-ssim']
parser = argparse.ArgumentParser(description='DMVC evaluation')

parser.add_argument('--pretrain', default = 'I:/DCVC_checkpoints/cvpr2023_video_psnr.pth.tar', help='Load pretrain model')
parser.add_argument('--gop_size', default = '32', type = int, help = 'The length of the gop')
parser.add_argument('--img_dir', default = 'I:/Data/HEVC/')
parser.add_argument('--intra_model',default= 'I:/DCVC_checkpoints/cvpr2023_image_psnr.pth.tar',  help = 'The intra coding method')
parser.add_argument('--cuda_id',default=0,type= int)
parser.add_argument('--target_bpp',default=0.13,type= float,help = 'average BPP of a single frame')
parser.add_argument('--sw',default=40 , type= int,help ='size of slide window')
parser.add_argument('--test_class', default = 'ClassC', type = str,  help = 'Choose from the test dataset')
parser.add_argument('--sec_id',default=0, type= int)
parser.add_argument('--pic_width',default=384, type= int)
parser.add_argument('--pic_height',default=192, type= int)
parser.add_argument('--write_stream',default=False, type= bool)
parser.add_argument('--stream_path', type=str, default="control_out_bin")


args = parser.parse_args()

device = torch.device('cuda', args.cuda_id)
if args.test_class == "ClassD":
    pic_width = 384
    pic_height = 192
elif args.test_class == "ClassC":
    pic_width = 832
    pic_height = 448
else:
    pic_width = 1920
    pic_height = 1024




test_dataset = CTS_con(args.img_dir, args.test_class, sec_id= args.sec_id)
test_loader = DataLoader(dataset = test_dataset, shuffle = False, num_workers = 0, batch_size = 1)

def eval_model(net,net2,y_q_scale_enc):
    

    c_,k_ = [0.08534 for i in range(4)],[-1.154 for i in range(4)]
    print("Evaluating...")
    net.eval()
    
    
    f = open('./data2.csv','w',encoding='utf-8')
    csv_writer = csv.writer(f)
    csv_writer.writerow(['gop_index', "BPP"])
    
    inter_cnt = 0
    intra_cnt = 0
    cnt = 0

    sum_psnr = 0

    
    sum_bpp = 0
    sum_bits = 0
    
    sum_intra_bpp = 0
    sum_inter_bpp = 0
    
    sum_intra_psnr = 0


    t0 = datetime.datetime.now()
    gop_bpp_target = 0.0
    gop_bpp = 0


    for batch_idx, (frames, intra_bpp, gop_size, i_frames) in enumerate(test_loader):
        batch_size, frame_length, _, _, _ = frames.shape
        h,w = pic_height,pic_width
        if args.gop_size > 0:
            gop_size = args.gop_size
        else:
            gop_size = gop_size.item()

        for frame_idx in tqdm(range(frame_length)):
            with torch.no_grad():
                
                if frame_idx % gop_size == 0:
                    if frame_length - frame_idx < args.sw:
                        sw = frame_length - frame_idx
                    else:
                        sw = args.sw
                    
                    gop_bpp_target = (args.target_bpp *(frame_idx + sw) - sum_bpp) * args.gop_size / sw                           
                    if frame_idx:
                        gop_bpp = 0.0
                        
                    x = frames[:,frame_idx].to(device)
                        ################################对图像进行padding,并输入网络#######                       
                    padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height,pic_width, 16)
                    x_padded = torch.nn.functional.pad(
                        x,
                        (padding_l, padding_r, padding_t, padding_b),
                        mode="replicate",
                    )                    
                    result = net2(x_padded,q_index =2)
                    ##############################参数计算#################################

                    x_hat = result["x_hat"].clamp(0., 1.)
                    x_hat = F.pad(x_hat, (-padding_l, -padding_r, -padding_t, -padding_b))
                    

                    intra_mse =  torch.mean((x_hat - frames[:, frame_idx ].to(device)).pow(2)) 
                    intra_psnr = torch.mean(10 * (torch.log(1. / intra_mse) / np.log(10))).cpu().detach().numpy()          
                    intra_bits = result["bit"]

                    csv_writer.writerow([str(frame_idx// gop_size),intra_bits / (batch_size * h * w)])
                    gop_bpp += intra_bits / (batch_size * h * w)

                    sum_bits += intra_bits
                    sum_bpp += intra_bits / (batch_size * h * w)
                    sum_intra_bpp += intra_bits / (batch_size * h * w)
                    
                    sum_psnr += intra_psnr
                    sum_intra_psnr += intra_psnr
                    cnt += 1
                    intra_cnt+=1
                    
                    dpb = {
                    "ref_frame": result["x_hat"],
                    "ref_feature": None,
                    "ref_mv_feature": None,
                    "ref_y": None,
                    "ref_mv_y": None,
                    }


                    continue

                x_curr = frames[:, frame_idx ].to(device) 
                x_padded = torch.nn.functional.pad(
                        x_curr,
                        (padding_l, padding_r, padding_t, padding_b),
                        mode="replicate",
                    )

                       
                n_index = frame_idx % args.gop_size

                left_rate = gop_bpp_target - gop_bpp

                if left_rate < 0: 
                    y_q_scale = 1/y_q_scale_enc[0]
                else:
                    
                    y_q_scale= R_s_gloabl(left_rate,c_, k_,n_index,y_q_scale_enc,args.gop_size)
                    
                q_step=float('{:.3f}'.format( 1 / y_q_scale))
                q_step = clamp(q_step, y_q_scale_enc[0],y_q_scale_enc[-1])
                
                q_index = np.log(q_step / y_q_scale_enc[0]) / np.log(y_q_scale_enc[-1] / y_q_scale_enc[0])
                result = net.encode_decode(x_padded, dpb, False, q_index, pic_height=pic_height, pic_width=pic_width,frame_idx=frame_idx % 4)
                
                dpb = result["dpb"]
                x_hat = dpb["ref_frame"]
                x_hat = x_hat.clamp_(0, 1)
                
                x_hat = F.pad(x_hat, (-padding_l, -padding_r, -padding_t, -padding_b))
                inter_mse = F.mse_loss(x_hat,x_curr)
                inter_psnr = 10 * (torch.log(1 * 1 / inter_mse) / np.log(10)).cpu().detach().numpy()
                bpp = result['bit']/ (batch_size * h * w)
                csv_writer.writerow([str(frame_idx// gop_size) , bpp ,c_[n_index%4]* y_q_scale**k_[n_index%4] ])
                if frame_idx <16:
                    c_[n_index %4], k_[n_index%4] = update_params_v2(bpp,c_[n_index%4], k_[n_index%4], torch.tensor(y_q_scale),10)
                elif frame_idx <160:
                    c_[n_index%4], k_[n_index%4] = update_params_v2(bpp,c_[n_index%4], k_[n_index%4], torch.tensor(y_q_scale),6)
                else:
                    c_[n_index%4], k_[n_index%4] = update_params_v2(bpp,c_[n_index%4], k_[n_index%4], torch.tensor(y_q_scale),2)          
                
             
                sum_bpp += bpp
                sum_inter_bpp += bpp
                gop_bpp += bpp
                inter_cnt +=1
                cnt += 1
                sum_psnr += inter_psnr



                    


                    
                

    t1 = datetime.datetime.now()
    deltatime = t1 - t0

    print("bpp:{:.6f} psnr:{:.4f}" \
          .format(sum_bpp / cnt, sum_psnr/cnt))

def check_dir_exist(check_dir):
    if not os.path.exists(check_dir):
        os.makedirs(check_dir)

def main():
    print(args)
#############################设置i帧模型###########
    i_state_dict = get_state_dict(args.intra_model)
    i_model = IntraNoAR(ec_thread= False,stream_part=1,inplace= True)
    i_model.load_state_dict(i_state_dict)
    i_model.to(device)
    i_model.eval()
#############################设置p帧模型###########    
    model = DMC(ec_thread=False,stream_part=1,inplace=True)
    model_dict = get_state_dict(args.pretrain)
    model.load_state_dict(model_dict)
    model.to(device)
    model.eval()
    num_params = 0
    if model is not None:
        model.update(force=True)
    i_model.update(force=True)
    for param in model.parameters():
        num_params += param.numel()
    print('The total number of the learnable parameters:', num_params)
####################设置量化步长##############
    if args.pretrain != '':
        print('Load the model from {}'.format(args.pretrain))
        print('Load the I-model from {}'.format(args.intra_model))
        y_q_scale_enc, y_q_scale_dec, mv_y_q_scale_enc, mv_y_q_scale_dec = DMC.get_q_scales_from_ckpt(args.pretrain)
        i_frame_q_scale_enc, i_frame_q_scale_dec = IntraNoAR.get_q_scales_from_ckpt(args.intra_model)
        
        print("y_q_scales_enc in I-frame ckpt: ", end='')
        for q in i_frame_q_scale_enc:
            print(f"{q:.3f}, ", end='')
        print()
        print("y_q_scales_dec in I-frame ckpt: ", end='')
        for q in i_frame_q_scale_dec:
            print(f"{q:.3f}, ", end='')
        print()         
        
        
        print("y_q_scale_enc in inter ckpt: ", end='')
        for q in y_q_scale_enc:
            print(f"{q:.3f}, ", end='')
        print()
        print("y_q_scale_dec in inter ckpt: ", end='')
        for q in y_q_scale_dec:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scale_enc in inter ckpt: ", end='')
        for q in mv_y_q_scale_enc:
            print(f"{q:.3f}, ", end='')
        print()
        print("mv_y_q_scale_dec in inter ckpt: ", end='')
        for q in mv_y_q_scale_dec:
            print(f"{q:.3f}, ", end='')
        print()
        y_q_scale = [float('{:.3f}'.format(i)) for i in y_q_scale_enc]

    eval_model(model, i_model,y_q_scale)

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    random.seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True

    main()
