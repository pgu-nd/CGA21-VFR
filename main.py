from model import TSR_R2UNet
import torch.nn as nn
import torch
from torch.nn import init
from torch.nn.modules import conv, Linear
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
import random
import fnmatch,os
import csv
from operator import itemgetter 
from math import pi

#os.environ["CUDA_VISIBLE_DEVICES"] = '2'

parser = argparse.ArgumentParser(description='PyTorch Implementation of FlowNet')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate')   # try different versions, such as 1e-2, 1e-3,1e-5
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--epochs', type=int, default=101, metavar='N',
                    help='number of epochs to train')

args = parser.parse_args()
print(not args.no_cuda)
print(torch.cuda.is_available())
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 10, 'pin_memory': True} if args.cuda else {}



batch_size = 20
test_batch_size =3
data_name = 'supernova'  

home_dir = '/afs/crc.nd.edu/user/p/pgu/Research/VFR_Experis/R2Net_supernova_REP_500_simple/'
result_dir = home_dir + 'result' 
model_dir = home_dir + 'Saved_model' 
Inference_dir = home_dir + 'inference' 


directory1 = os.path.dirname(result_dir)
if not os.path.exists(directory1):
    os.makedirs(directory1)
directory2 = os.path.dirname(model_dir)
if not os.path.exists(directory2):
    os.makedirs(directory2)
directory3 = os.path.dirname(Inference_dir)
if not os.path.exists(directory3):
    os.makedirs(directory3)



def Load_Data(filename, batch_size, x_dim,y_dim,z_dim,data_name): # [B, 3, 128, 128, 128], B is the data you want to deal with
    ### simply append B number of data together
    #num_files = len(fnmatch.filter(os.listdir(filename),'*.txt'))
    #print(num_files)
    #randomList = random.sample(range(1, num_files+1), batch_size)
    randomList = [1,2,3,4,5,6,7,8,9,11,12,13,14,15,16,17,18,19,20,21]
    print('randomList', randomList)
    points_concat = []
    vecs_concat = []

    h=0
    GVF_concat = np.zeros((batch_size,3,x_dim,y_dim,z_dim))
    gt_concat = np.zeros((batch_size,3,x_dim,y_dim,z_dim))
    for i in randomList:
        streamline_path = filename + '/' + str(data_name) + '-500-'+ '{:03d}'.format(i) +'.txt'
        print('streamline_path', streamline_path)
        # read the file: supernova-rep-005.txt
        streamline = open(streamline_path,'rb')
        
        ## the first line is the points positions
        s = str(streamline.readline(),'utf-8').strip()
        point = s.split('\t')
        points=[]
        for p in point :
            points.append(float(p))
        #print('points length', len(points))
        points_concat.append(points)

        ## the second line is the points' velocity in the streamlines
        s1 = str(streamline.readline(),'utf-8').strip()
        #print('test', s.isspace()) 
        vec = s1.split("\t")
        vecs = []
        for v in vec :
            vecs.append(float(v))
        #print('vecs length', len(vecs))
        vecs_concat.append(vecs)

        ####### read GVF files, [B, 3, 128, 128, 128]
        GVF_path = filename + '/' + str(data_name) + '-500-gvf-'+ '{:03d}'.format(i) +'.vec'
        print('GVF_path', GVF_path)
        v = np.fromfile(GVF_path,dtype='<f')
        #print('v', v) 
        #print('v shape', v.shape)
        v_reshape = v.reshape(z_dim,y_dim,x_dim,3,1).transpose()
        #print('v ', v)
        #print('v shape', v.shape)
        GVF_concat[h,:,:,:,:] = v_reshape
        #print('GVF_concat ', GVF_concat)
        #print('GVF_concat shape', GVF_concat.shape)


        #### read gt, [B, 3, 128, 128, 128]
        gt_path = '/afs/crc.nd.edu/user/p/pgu/Research/VFR_Experis/Data_REP_300_supernova/gt/' + str(data_name) + '{:03d}'.format(i) +'.vec'
        #print('gt_path', gt_path)
        v1 = np.fromfile(gt_path,dtype='<f')
        #print('v1', v1) 
        #print('v1 shape', v1.shape)
        v1_reshape = v1.reshape(z_dim,y_dim,x_dim,3,1).transpose()
        #print('v1', v1)
        #print('v1 shape', v1.shape)
        gt_concat[h,:,:,:,:] = v1_reshape
        #print('gt_concat', gt_concat)
        #print('gt_concat shape', gt_concat.shape)
        h = h+1
    
    GVF_concat_input = torch.FloatTensor(GVF_concat)
    #print('GVF_concat', GVF_concat.shape) # (10, 3, 128, 128, 128)
    #print('GVF batch size', GVF_concat.shape[0]) # 10
    

    ############ concat [0,1,2],  [1,2,3], ..., [7,8,9] and save to GVF_concat_new
    GVF_concat_new = []
    for i in range(1,  GVF_concat.shape[0]-1): # 1,2,3,4,5,6,7,8
        GVF_concat_three = torch.cat((GVF_concat_input[i-1],GVF_concat_input[i],GVF_concat_input[i+1]),0)
        #print('GVF_concat_three shape', GVF_concat_three.shape)
        GVF_concat_new.append(GVF_concat_three)
    GVF_concat_new = torch.stack(GVF_concat_new,0)
    #print('GVF_concat_new_ shape', GVF_concat_new.shape) # torch.Size([8, 9, 128, 128, 128])
    ############


    ## save  concat GVF and gt files
    ## here need to remove the first and last elements
    GVF_concat = np.asarray(GVF_concat[1:batch_size-1,:,:,:,:],dtype='<f')
    #print('GVF_concat shape', len(GVF_concat)) # 8
    GVF_concat = GVF_concat.flatten('F')
    GVF_concat.tofile(result_dir + '/500_coarse_GVF.vec',format='<f')

    gt_concat = np.asarray(gt_concat[1:batch_size-1,:,:,:,:],dtype='<f')
    #print('gt_concat before flatten', gt_concat)
    #print('gt_concat shape', len(gt_concat)) # 8
    gt_concat = gt_concat.flatten('F')
    #print('gt_concat', gt_concat)
    gt_concat.tofile(result_dir + '/500_gt.vec',format='<f')
    #################### 

    vecs_concat = np.array(vecs_concat) # [[], []]  
    # print('vecs_concat shape', len(vecs_concat), 
    #     len(vecs_concat[0]), 
    #     len(vecs_concat[1]))
    points_concat = np.array(points_concat) # [[], []]
    # print('points_concat shape', len(points_concat), 
    #     len(points_concat[0]), 
    #     len(points_concat[1]))
    streamline_array_concat = []
    vecs_array_concat = []
    #print(len(vecs_concat)) # 10
    ### Need to remove the first and last points' velocity
    for j in range(1, len(vecs_concat)-1): # 1, ...,7, 8
        vec_array = []
        streamline_array = []

        num_posi = int(len(vecs_concat[j])/3) 
        #print(num_posi)  #17194
        ### read the points and velocity as array [[x1,y1,z1],[],...,[]]
        for i in range(num_posi+1): # 0,1,..., num_posi
            if i != num_posi:
                streamline_array.append((points_concat[j][3*i:3*(i+1)]))
                vec_array.append((vecs_concat[j][3*i:3*(i+1)]))
            else:
                if (len(points_concat[j][3*i:])>0):
                    print('test')
                    streamline_array.append((points_concat[j][3*i:]))
                    vec_array.append((vecs_concat[j][3*i:]))


        streamline_array_concat.append(streamline_array)
        vecs_array_concat.append(vec_array)
    
    # print('streamline_array_concat shape', 
    #         len(streamline_array_concat), 
    #         len(streamline_array_concat[0]), 
    #         len(streamline_array_concat[0][0]),
    #         len(streamline_array_concat[1]), 
    #         len(streamline_array_concat[1][0]),)
    # print('vecs_array_concat shape', 
    #         len(vecs_array_concat), 
    #         len(vecs_array_concat[0]), 
    #         len(vecs_array_concat[0][0]),
    #         len(vecs_array_concat[1]), 
    #         len(vecs_array_concat[1][0]),)
    # [[[x,y,z], ..., [x,y,z]], [[x,y,z], ..., [x,y,z]]] batch size =2
    return streamline_array_concat,vecs_array_concat, GVF_concat_new



def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        m.weight.data.normal_(0,0.01)
    elif classname.find("Linear")!=-1:
        m.weight.data.normal_(0,0.01)
    elif classname.find("BatchNorm")!=-1:
        m.weight.data.normal_(1.0,0.01)
        #m.bias.data.constant_(0.0)

def weight_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find("Conv")!=-1:
        init.kaiming_normal(m.weight.data,a=0,mode="fan_in")
    elif classname.find("Linear")!=-1:
        init.kaiming_normal(m.weight.data,a=0,mode="fan_in")
    elif classname.find("BatchNorm")!=-1:
        init.normal(m.weight.data,1.0,0.02)
        init.constant(m.bias.data,0.0)


class parm():
    def __init__(self,max_streamline=3000,min_point=3,max_point=1000,segment_length=1.0,trace_interval=0.2,max_step=500):
        self.max_streamline = max_streamline
        self.min_point = min_point
        self.max_point = max_point
        self.segment_length = segment_length
        self.trace_interval = trace_interval
        self.max_step = max_step


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 40))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def getPSNR(t,v):
    m = np.max(t)-np.min(t)
    mse = np.mean((t-v)**2) 
    #print('mse', mse)
    psnr = 20*np.log10(m)-10*np.log10(mse)
    return psnr

def getAAD(t,v):
    t = torch.FloatTensor(t)
    v = torch.FloatTensor(v)
    cos = torch.sum(t*v,dim=0) / (torch.norm(t, dim=0) * torch.norm(v, dim=0) + 1e-10)
    cos[cos>1] = 1
    cos[cos<-1] = -1
    aad = torch.mean(torch.acos(cos)).item() / pi
    return aad

def coverage(streamlines,vf_w,vf_h,vf_d):
    #streamlines N*3
    cube = np.zeros((vf_w,vf_h,vf_d))
    for j in range(0,len(streamlines)):
        x = int(streamlines[j][0])
        y = int(streamlines[j][1])
        z = int(streamlines[j][2])
        cube[x][y][z] = 1
        cube[x+1][y][z] = 1
        cube[x][y+1][z] = 1
        cube[x][y][z+1] = 1
        cube[x+1][y+1][z] = 1
        cube[x+1][y][z+1] = 1
        cube[x][y+1][z+1] = 1
        cube[x+1][y+1][z+1] = 1
    return coverage

def get_streamlines_and_velocities(streamlines,velocities,coverage,vf_w,vf_h,vf_d):
    p = np.random.permutation(len(streamlines))
    #print('p', p)
    streamlines_copy = list(itemgetter(*p)(streamlines)) #streamlines[p]
    velocities_copy = list(itemgetter(*p)(velocities)) #velocities[p]
    state = []
    points = []
    velos = []
    cube = np.zeros((vf_w,vf_h,vf_d))
    for j in range(0,len(streamlines_copy)):
        x = int(streamlines_copy[j][0])
        y = int(streamlines_copy[j][1])
        z = int(streamlines_copy[j][2])
        if [x,y,z] not in state:
            cube[x][y][z] = 1
            cube[x+1][y][z] = 1
            cube[x][y+1][z] = 1
            cube[x][y][z+1] = 1
            cube[x+1][y+1][z] = 1
            cube[x+1][y][z+1] = 1
            cube[x][y+1][z+1] = 1
            cube[x+1][y+1][z+1] = 1
            state.append([x,y,z])
            points.append(streamlines_copy[j])
            velos.append(velocities_copy[j])
        if j%10000==0:
            comparsion = cube == coverage
            if comparsion.all():
                return points, velos
    return points, velos

    
def train(epochs,gvf_VFD,name):
    vector_field_concat = np.zeros((batch_size-2,3,vf_w,vf_h,vf_d))
    for itera in range(1,epochs+1):
        loss = 0
        x = time.time()
        print("========================")
        print('iter ', itera)
        for batch in range(0, batch_size-2):  # 0,1,2,..., 7
            print('batch', batch)
            length = len(streamline_array_concat[batch])
            print('totla number points ', length)
            # sample_points = int(length*0.5)
            # Num = int(len(streamline_array_concat[batch])/sample_points)  #int(67.16)
            # print('split Num parts ', Num)  # 67
            x1 = time.time()
            time1 = time.time()
            cover = coverage(streamline_array_concat[batch],vf_w,vf_h,vf_d)
            points, velos = get_streamlines_and_velocities(streamline_array_concat[batch],vecs_array_concat[batch],cover,vf_w,vf_h,vf_d)
            time2 = time.time()
            print('sample points cost', str(time2-time1))
            print('valid points number', len(points))

            positions_groud =torch.FloatTensor(points)
            vec_ground = torch.FloatTensor(velos)

            # ### read the points and velocity as array [[x1,y1,z1],[],...,[]]
            # for batch_idx in range(Num+1): # 0,1,..., Num
            #     if batch_idx != Num:
            #         positions_groud = torch.FloatTensor(streamline_array_concat[batch][sample_points*batch_idx:sample_points*(batch_idx+1)])
            #         vec_ground = torch.FloatTensor(vecs_array_concat[batch][sample_points*batch_idx:sample_points*(batch_idx+1)])
            #     else:
            #         if (len(streamline_array_concat[batch][sample_points*batch_idx:])>0):
            #             positions_groud = torch.FloatTensor(streamline_array_concat[batch][sample_points*batch_idx:])
            #             vec_ground = torch.FloatTensor(vecs_array_concat[batch][sample_points*batch_idx:])

            #     # sample_points = 9000 #int(length*0.1) # sampled 20% random points
            #     # #print('sample_points ', sample_points)
            #     # randomPoint_List = random.sample(range(0, length), sample_points)
            #     # #print('randomPoint_List ', randomPoint_List)

            #     # ####### test 
            #     # # randomPoint_List = [0, 3]
            #     # # print(streamline_array_concat[batch][0])
            #     # # print(streamline_array_concat[batch][3])
            #     # # print(streamline_array_concat[batch][0:4])

            #     # # print(vecs_array_concat[batch][0])
            #     # # print(vecs_array_concat[batch][3])
            #     # # print(vecs_array_concat[batch][0:4])
            #     # #######

            #     # streamline_array_concat_list = list(itemgetter(*randomPoint_List)(streamline_array_concat[batch])) 
            #     # vecs_array_concat_list = list(itemgetter(*randomPoint_List)(vecs_array_concat[batch])) 

            #     # positions_groud =torch.FloatTensor(streamline_array_concat_list)
            #     # vec_ground = torch.FloatTensor(vecs_array_concat_list)

            #     #print('positions_groud', positions_groud)
            #     #print('vec_ground', vec_ground)
                
                
            #print('gvf_VFD',gvf_VFD)
            #print('gvf_VFD shape', gvf_VFD.shape) #torch.Size([8, 9, 128, 128, 128])
            gvf_VFD_batch = gvf_VFD[batch,:,:,:,:]
            #print('gvf_VFD_batch',gvf_VFD_batch)
            #print('gvf_VFD_batch shape',gvf_VFD_batch.shape)  #torch.Size([9, 128, 128, 128])
            gvf_VFD_batch_epand = torch.unsqueeze(gvf_VFD_batch, 0)
            #print('gvf_VFD_batch_epand',gvf_VFD_batch_epand)
            #print('gvf_VFD_batch_epand shape', gvf_VFD_batch_epand.shape) #torch.Size([1, 9, 128, 128, 128])

            
            if args.cuda:
                positions_groud = positions_groud.cuda()
                gvf_VFD_batch_epand = gvf_VFD_batch_epand.cuda()
                vec_ground = vec_ground.cuda()
            positions_groud = Variable(positions_groud)
            gvf_VFD_batch_epand = Variable(gvf_VFD_batch_epand)
            vec_ground = Variable(vec_ground)

            #print('gvf_VFD_batch_epand shape', gvf_VFD_batch_epand.shape) #torch.Size([1, 3, 128, 128, 128])
            #print('positions_groud', positions_groud)
            #print('positions_groud shape', positions_groud.shape) # torch.Size([256, 3])
            #print('vec_ground', vec_ground)
            #print('vec_ground shape', vec_ground.shape) # torch.Size([sample_points, 3])

            ####################################
            # size1= vec_ground.shape[0]
            # size2= vec_ground.shape[1]
            # vec_ground_test_flatten = torch.flatten(vec_ground)
            # print('vec_ground_test_flatten', vec_ground_test_flatten)
            # vec_ground_test_reshape = vec_ground_test_flatten.reshape(size1, size2)
            # print('vec_ground_test_reshape', vec_ground_test_reshape)
            # print('Check reshape', vec_ground-vec_ground_test_reshape)
            ####################################

            vector_field,vecs = model(gvf_VFD_batch_epand,positions_groud)
            #print('vector_field',vector_field)
            #print('vector_field shape',vector_field.shape)
            vector_field = vector_field.cpu().detach().numpy()
            #print('vector_field',vector_field)
            #print('vector_field shape',vector_field.shape)
            
            vector_field_concat[batch,:,:,:,:] = vector_field
            #print('vector_field_concat', vector_field_concat)
            #print('vector_field_concat shape', vector_field_concat.shape)
            #print('vecs', vecs)
            #print('vecs shape', vecs.shape) # torch.Size([sample points, 3])
            optimizer.zero_grad()
            loss_fun = nn.MSELoss(size_average=False)
            loss_g = loss_fun(vecs,vec_ground)
            loss = loss+loss_g.item()
            loss_g.backward()
            optimizer.step()
            x2 = time.time()
            print('Process one data:'+str(x2-x1))

        y = time.time()
        print("loss = "+str(loss))
        print("Time = "+str(y-x))
        with open("Loss.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=["Epochs", "Loss", "Time"])
                writer.writeheader()
                writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow([itera, loss, y-x])
        if itera%40==0:
            adjust_learning_rate(optimizer,itera)
        if itera%10==0 or itera==1:

            data = np.asarray(vector_field_concat,dtype='<f')
            #print('data', data)
            #print('data shape before', data.shape)
            data = data.flatten('F')
            #print('data after flatten ', data)
            #print('data shape after', data.shape)
            data.tofile(result_dir + '/'+ str(name)+'-epochs-'+ str(itera) + '.vec',format='<f')
            
            #saveVectorfield(vector_field,num,name)
            torch.save(model, model_dir + '/' + str(name) + '-REP-500-epochs-'+ str(itera) + '-vec-model.pth')
            

            gt_path = result_dir + '/500_gt.vec'
            gt = np.fromfile(gt_path,dtype='<f')

            result_path = result_dir + '/' + str(name) + '-epochs-' + str(itera) + '.vec'
            result = np.fromfile(result_path,dtype='<f')

            gvf_path = result_dir + '/500_coarse_GVF.vec'
            gvf = np.fromfile(gvf_path,dtype='<f')
            
            gvf_PSNR = getPSNR(gt,gvf)
            #gt_flatten = gt.flatten('F')
            res_PSNR = getPSNR(gt,result)
            print('PSNR result')
            print(str(name)+'-epochs-'+ str(itera) + '.vec')
            print('res_PSNR is: ', res_PSNR)
            print(str(name)+ 'coarse_GVF' + '.vec')
            print('gvf_PSNR is: ', gvf_PSNR)

            with open("PSNR_result.csv", "a") as f:
                writer = csv.DictWriter(f, fieldnames=["Epochs", "Result","Coarse_GVF"])
                writer.writeheader()
                writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
                writer.writerow([itera, res_PSNR, gvf_PSNR])

############################################# Inference data ########################
def Load_TestData(filename, test_batch_size, x_dim,y_dim,z_dim,testData_List,data_name): # [3, 3, 128, 128, 128], B is the data you want to deal with
    list = [testData_List]
    print('list', list)
    points_concat = []
    vecs_concat = []

    h=0
    GVF_concat = np.zeros((test_batch_size,3,x_dim,y_dim,z_dim))
    gt_concat = np.zeros((test_batch_size,3,x_dim,y_dim,z_dim))
    for i in list[0]:
        streamline_path = filename + '/' + str(data_name) + '-500-'+ '{:03d}'.format(i) +'.txt'
        print('streamline_path', streamline_path)
        # read the file: supernova-rep-005.txt
        streamline = open(streamline_path,'rb')
        
        ## the first line is the points positions
        s = str(streamline.readline(),'utf-8').strip()
        point = s.split('\t')
        points=[]
        for p in point :
            points.append(float(p))
        #print('points length', len(points))
        points_concat.append(points)

        ## the second line is the points' velocity in the streamlines
        s1 = str(streamline.readline(),'utf-8').strip()
        #print('test', s.isspace()) 
        vec = s1.split("\t")
        vecs = []
        for v in vec :
            vecs.append(float(v))
        #print('vecs length', len(vecs))
        vecs_concat.append(vecs)

        ####### read GVF files, [B, 3, 128, 128, 128]
        GVF_path = filename + '/' +str(data_name)+ '-500-gvf-'+ '{:03d}'.format(i) +'.vec'
        print('GVF_path', GVF_path)
        v = np.fromfile(GVF_path,dtype='<f')
        #print('v', v) 
        #print('v shape', v.shape)
        v_reshape = v.reshape(z_dim,y_dim,x_dim,3,1).transpose()
        #print('v ', v)
        #print('v shape', v.shape)
        GVF_concat[h,:,:,:,:] = v_reshape
        #print('GVF_concat ', GVF_concat)
        #print('GVF_concat shape', GVF_concat.shape)


        #### read gt, [B, 3, 128, 128, 128]
        gt_path = '/afs/crc.nd.edu/user/p/pgu/Research/VFR_Experis/Data_REP_300_supernova/gt/' +str(data_name) + '{:03d}'.format(i) +'.vec'
        #print('gt_path', gt_path)
        v1 = np.fromfile(gt_path,dtype='<f')
        #print('v1', v1) 
        #print('v1 shape', v1.shape)
        v1_reshape = v1.reshape(z_dim,y_dim,x_dim,3,1).transpose()
        #print('v1', v1)
        #print('v1 shape', v1.shape)
        gt_concat[h,:,:,:,:] = v1_reshape
        #print('gt_concat', gt_concat)
        #print('gt_concat shape', gt_concat.shape)
        h = h+1
    
    GVF_concat_input = torch.FloatTensor(GVF_concat)
    #print('GVF_concat', GVF_concat.shape) #(3, 3, 128, 128, 128)

    GVF_concat_new = []
    for i in range(1,  GVF_concat.shape[0]-1): # 1,2,3,4,5,6,7,8
        GVF_concat_three = torch.cat((GVF_concat_input[i-1],GVF_concat_input[i],GVF_concat_input[i+1]),0)
        #print('GVF_concat_three shape', GVF_concat_three.shape)
        GVF_concat_new.append(GVF_concat_three)
    GVF_concat_new = torch.stack(GVF_concat_new,0)
    #print('GVF_concat_new_ shape', GVF_concat_new.shape) # torch.Size([1, 9, 128, 128, 128])

    ## save  concat GVF and gt files
    ## here need to remove the first and last elements
    GVF_concat = np.asarray(GVF_concat[1:test_batch_size-1,:,:,:,:],dtype='<f')
    #print('GVF_concat shape', len(GVF_concat)) # 1
    GVF_concat = GVF_concat.flatten('F')
    GVF_concat.tofile(Inference_dir+ '/'+str(data_name)+ '{:03d}'.format(list[0][1])+ 'coarse_GVF.vec',format='<f')

    gt_concat = np.asarray(gt_concat[1:test_batch_size-1,:,:,:,:],dtype='<f')
    #print('gt_concat before flatten', gt_concat)
    gt_concat = gt_concat.flatten('F')
    #print('gt_concat', gt_concat)
    gt_concat.tofile(Inference_dir + '/'+str(data_name) + '{:03d}'.format(list[0][1])+ 'gt.vec',format='<f')


    vecs_concat = np.array(vecs_concat) # [[], []]  
    # print('vecs_concat shape', len(vecs_concat), 
    #     len(vecs_concat[0]), 
    #     len(vecs_concat[1]))

    points_concat = np.array(points_concat) # [[], []]
    # print('points_concat shape', len(points_concat), 
    #     len(points_concat[0]), 
    #     len(points_concat[1]))

    streamline_array_concat = []
    vecs_array_concat = []

    for j in range(1, len(vecs_concat)-1):
        vec_array = []
        streamline_array = []

        num_posi = int(len(vecs_concat[j])/3) 
        #print(num_posi)  #17194
        ### read the points and velocity as array [[x1,y1,z1],[],...,[]]
        for i in range(num_posi+1): # 0,1,..., num_posi
            if i != num_posi:
                streamline_array.append((points_concat[j][3*i:3*(i+1)]))
                vec_array.append((vecs_concat[j][3*i:3*(i+1)]))
            else:
                if (len(points_concat[j][3*i:])>0):
                    print('test')
                    streamline_array.append((points_concat[j][3*i:]))
                    vec_array.append((vecs_concat[j][3*i:]))


        streamline_array_concat.append(streamline_array)
        vecs_array_concat.append(vec_array)
    
    # print('streamline_array_concat shape', 
    #         len(streamline_array_concat), 
    #         len(streamline_array_concat[0]), 
    #         len(streamline_array_concat[0][0]),
    #         len(streamline_array_concat[1]), 
    #         len(streamline_array_concat[1][0]),)
    # print('vec_array_concat shape', 
    #         len(vec_array_concat), 
    #         len(vec_array_concat[0]), 
    #         len(vec_array_concat[0][0]),
    #         len(vec_array_concat[1]), 
    #         len(vec_array_concat[1][0]),)
    # [[[x,y,z], ..., [x,y,z]], [[x,y,z], ..., [x,y,z]]] batch size =2
    return streamline_array_concat,vecs_array_concat, GVF_concat_new

def test(model,gvf_VFD,name, test_num):
    x = time.time()
    if args.cuda:
        gvf_VFD = gvf_VFD.cuda()
    vector_field = model(gvf_VFD, None)
    #print('vector_field',vector_field)
    #print('vector_field shape',vector_field.shape)
    vector_field = vector_field.cpu().detach().numpy()
    #print('vector_field',vector_field)
    #print('vector_field shape',vector_field.shape) 
    y = time.time()
    print('Time: ', str(y-x))  
    data = np.asarray(vector_field,dtype='<f')
    data = data.flatten('F')
    data.tofile(Inference_dir + '/result/'+ str(name) + '{:03d}'.format(test_num)+'-REPR2Net-500-vec.vec',format='<f')

    gt_path = Inference_dir + '/' + str(name) + '{:03d}'.format(test_num)+ 'gt.vec'
    gt = np.fromfile(gt_path,dtype='<f')

    result_path = Inference_dir +'/result/'+str(name)+'{:03d}'.format(test_num)+'-REPR2Net-500-vec.vec'
    result = np.fromfile(result_path,dtype='<f')

    gvf_path = Inference_dir +'/'+ str(name)+ '{:03d}'.format(test_num)+ 'coarse_GVF.vec'
    gvf = np.fromfile(gvf_path,dtype='<f')
    
    gvf_PSNR = getPSNR(gt,gvf)
    res_PSNR = getPSNR(gt,result)
    print('Inference PSNR result')
    print('res_PSNR is: ', res_PSNR)
    print('gvf_PSNR is: ', gvf_PSNR)
    gvf_AAD = getAAD(gt,gvf)
    res_AAD = getAAD(gt,result)
    print('Inference AAD result')
    print('gvf_AAD is: ', gvf_AAD)
    print('res_AAD is: ', res_AAD)

    with open("Inference_PSNR_result.csv", "a") as f:
        writer = csv.DictWriter(f, fieldnames=["test_file","Result_PSNR","Coarse_GVF_PSNR", "Time", "Result_AAD","Coarse_GVF_AAD",])
        writer.writeheader()
        writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL, lineterminator='\n')
        writer.writerow([test_num,res_PSNR, gvf_PSNR, y-x, res_AAD, gvf_AAD])

#####################################################################

vf_w = 128
vf_h = 128
vf_d = 128
parm = parm()

#####################  train the model 
data_dir = '/afs/crc.nd.edu/group/vis/PaciVis21/Data/REP/Data_supernova/500'
streamline_array_concat,vecs_array_concat, GVF_concat_input = Load_Data(data_dir,batch_size,vf_w,vf_h,vf_d,data_name)
print(str(batch_size)+ ' set of streamlines' + " total length is " + str(len(streamline_array_concat)))
print('GVF_concat_input shape', GVF_concat_input.shape)
#### load the model
model = TSR_R2UNet(3,1,16) 
if args.cuda:
    model.cuda()
model.apply(weights_init_normal) # try to use weight_init_kaiming() to initialize weights 
optimizer = optim.Adam(model.parameters(), lr=args.lr)
#### train the model 
train(args.epochs,GVF_concat_input, data_name)





#####################  inference the model to new data
# testdata_dir = '/afs/crc.nd.edu/group/vis/PaciVis21/Data/REP/Data_supernova/500'
# rest_List = [
# 1,1,2,3,4,5,6,7,8,9,10,
# 11,12,13,14,15,16,17,18,19,20,
# 21,22,23,24,25,27,26,28,29,30,
# 31,32,33,34,35,36,37,38,39,40,
# 41,42,43,44,45,46,47,48,49,50,
# 51,52,53,54,55,56,57,58,59,
# 61,62,63,64,65,66,67,68,69,70,
# 71,72,73,74,75,76,77,78,79,80,
# 81,82,83,84,85,86,87,88,89,90,
# 91,92,93,94,95,96,97,98,99,100,100
# ]
# for i in range(1, len(rest_List)-1):
#     testData_List = [rest_List[i-1], rest_List[i], rest_List[i+1]]
#     #print('testData_List',testData_List)
#     #print(rest_List[i])
#     ##### load the saved trained model 
#     streamline_array_concat,vecs_array_concat, GVF_concat_input = Load_TestData(testdata_dir,test_batch_size,vf_w,vf_h,vf_d, testData_List, data_name)
#     model = torch.load('/afs/crc.nd.edu/user/p/pgu/Research/VFR_Experis/R2Net_supernova_REP_500_simple/Saved_model' + '/supernova-REP-500-epochs-100-vec-model.pth',map_location=lambda storage, loc:storage)
#     if args.cuda:
#         model.cuda()
#     ##### applied the trained model on new data
#     test(model,GVF_concat_input,data_name,rest_List[i])





