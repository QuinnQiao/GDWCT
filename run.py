import torch
import torch.nn as nn
import torch.optim as optim
from torch import cuda
from data_loader import *
from model import *
import time
import datetime
import os, sys
from utils.util import *
from torch.backends import cudnn
from scipy.linalg import block_diag
import torchvision.utils as vutils
import torchvision.transforms as T
from PIL import Image


class Run(object):
    def __init__(self, config):
        self.data_loader = get_loader(config['DATA_PATH'],
                                    crop_size=config['CROP_SIZE'], resize=config['RESIZE'], 
                                    batch_size=config['BATCH_SIZE'], dataset=config['DATASET'], 
                                    mode=config['MODE'], num_workers=config['NUM_WORKERS'])

        self.config = config
        self.device = torch.device("cuda:%d" % (int(config['GPU1'])) if torch.cuda.is_available() else "cpu")
        print(self.device)

        C = self.config['G']['CONTENT_DIM'] # The number of channels of the content feature.
        n_mem = C // self.config['N_GROUP'] # The number of blocks in the coloring matrix: G, The number of elements for each block: n_members^2
        self.mask = self.get_block_diagonal_mask(n_mem) # This is used in generators to make the coloring matrix the block diagonal form.

        self.make_dir()
        self.init_network()
        self.loss = {}

        print(config)

        if config['LOAD_MODEL']:
            self.load_pretrained_model(self.config['START'])

    def get_block_diagonal_mask(self, n_member):
        G = self.config['N_GROUP']
        ones = np.ones((n_member,n_member)).tolist()
        mask = block_diag(ones,ones)
        for i in range(G-2):
            mask = block_diag(mask,ones)
        return torch.from_numpy(mask).to(self.device).float()

    def make_dir(self):
        if not os.path.exists(self.config['MODEL_SAVE_PATH']):
            os.makedirs(self.config['MODEL_SAVE_PATH'])

    def init_network(self):
        """Create a generator and a discriminator."""
        G_opts = self.config['G']
        D_opts = self.config['D']
        
        self.G_A = Generator(G_opts['FIRST_DIM'], G_opts['N_RES_BLOCKS'], self.mask, self.config['N_GROUP'],
                           G_opts['MLP_DIM'], G_opts['BIAS_DIM'], G_opts['CONTENT_DIM'], self.device)
        self.G_B = Generator(G_opts['FIRST_DIM'], G_opts['N_RES_BLOCKS'], self.mask, self.config['N_GROUP'],
                           G_opts['MLP_DIM'], G_opts['BIAS_DIM'], G_opts['CONTENT_DIM'], self.device)

        G_params = list(self.G_A.parameters()) + list(self.G_B.parameters()) # + list(blah)
        self.G_optimizer = torch.optim.Adam([p for p in G_params if p.requires_grad], self.config['G_LR'], [self.config['BETA1'], self.config['BETA2']], weight_decay=self.config['WEIGHT_DECAY'])
        self.G_scheduler = get_scheduler(self.G_optimizer, config)
        self.G_A.apply(weights_init(self.config['INIT']))
        self.G_B.apply(weights_init(self.config['INIT']))

        if self.config['MODE'] == 'train':
            self.D_A = Discriminator(3, D_opts)
            self.D_B = Discriminator(3, D_opts)
            D_params = list(self.D_A.parameters()) + list(self.D_B.parameters())
            self.D_optimizer = torch.optim.Adam([p for p in D_params if p.requires_grad], self.config['D_LR'], [self.config['BETA1'], self.config['BETA2']], weight_decay=self.config['WEIGHT_DECAY'])
            self.D_scheduler = get_scheduler(self.D_optimizer, config)
            self.D_A.apply(weights_init('gaussian'))
            self.D_B.apply(weights_init('gaussian'))
        
        # print_network(self.G, 'G')
        # print_network(self.D, 'D')

        self.set_gpu()

    def set_gpu(self):
        def multi_gpu(gpu1, gpu2, model):
            model = nn.DataParallel(model, device_ids=[gpu1, gpu2])
            return model

        gpu1 = int(self.config['GPU1'])
        gpu2 = int(self.config['GPU2'])
        if self.config['DATA_PARALLEL']:
            self.G_A = multi_gpu(gpu1, gpu2, self.G_A)
            self.G_B = multi_gpu(gpu1, gpu2, self.G_B)
            if self.config['MODE'] == 'train':
                self.D_A = multi_gpu(gpu1, gpu2, self.D_A)
                self.D_B = multi_gpu(gpu1, gpu2, self.D_B)

        self.G_A.to(self.device)
        self.G_B.to(self.device)
        if self.config['MODE'] == 'train':
            self.D_A.to(self.device)
            self.D_B.to(self.device)

    def l1_criterion(self, input, target):
        return torch.mean(torch.abs(input - target))

    def reg(self, x_arr):
        # whitening_reg: G,C//G,C//G
        I = torch.eye(x_arr[0][0].size(1)).unsqueeze(0).to(self.device) # 1,C//G,C//G
        loss = torch.FloatTensor([0]).to(self.device)
        for x in x_arr:
            x = torch.cat(x,dim=0) # G*(# of style),C//G,C//G
            loss = loss + torch.mean(torch.abs(x-I))
        return loss / len(x_arr)

    def model_save(self, iteration):
        self.G_A = self.G_A.cpu()
        self.G_B = self.G_B.cpu()
        self.D_A = self.D_A.cpu()
        self.D_B = self.D_B.cpu()

        torch.save(self.G_A.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'G_A_%s_%d.pth' % (self.config['SAVE_NAME'],iteration)))
        torch.save(self.G_B.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'G_B_%s_%d.pth' % (self.config['SAVE_NAME'],iteration)))
        torch.save(self.D_A.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'D_A_%s_%d.pth' % (self.config['SAVE_NAME'],iteration)))
        torch.save(self.D_B.state_dict(),
            os.path.join(self.config['MODEL_SAVE_PATH'], 'D_B_%s_%d.pth' % (self.config['SAVE_NAME'],iteration)))
        
        self.set_gpu()

    def load_pretrained_model(self, iteration):
        self.G_A.load_state_dict(torch.load(os.path.join(
            self.config['MODEL_SAVE_PATH'], 'G_A_%s_%d.pth' % (self.config['SAVE_NAME'], iteration)), map_location=self.device))
        self.G_B.load_state_dict(torch.load(os.path.join(
            self.config['MODEL_SAVE_PATH'], 'G_B_%s_%d.pth' % (self.config['SAVE_NAME'], iteration)), map_location=self.device))
        if self.config['MODE'] == 'train':
            self.D_A.load_state_dict(torch.load(os.path.join(
                self.config['MODEL_SAVE_PATH'], 'D_A_%s_%d.pth' % (self.config['SAVE_NAME'], iteration)), map_location=self.device))
            self.D_B.load_state_dict(torch.load(os.path.join(
                self.config['MODEL_SAVE_PATH'], 'D_B_%s_%d.pth' % (self.config['SAVE_NAME'], iteration)), map_location=self.device))

    def update_learning_rate(self):
        if self.G_scheduler is not None:
            self.G_scheduler.step()
        if self.D_scheduler is not None:
            self.D_scheduler.step()

    def train_ready(self):
        self.G_A.train()
        self.G_B.train()
        self.D_A.train()
        self.D_B.train()

    def test_ready(self):
        self.G_A.eval()
        self.G_B.eval()

    def clamping_alpha(self,G):
        for gdwct in G.decoder.gdwct_modules:
            gdwct.alpha.data.clamp_(0,1)

    def get_latent(self, x, domain='A'):
        if domain == 'A':
            G = self.G_A.module if self.config['DATA_PARALLEL'] else self.G_A
        else:
            G = self.G_B.module if self.config['DATA_PARALLEL'] else self.G_B

        self.clamping_alpha(G)

        c = G.c_encoder(x)
        s = G.s_encoder(x, debug=True)

        return c, s

    def update_G(self, x_A, x_B, isTrain=True):
        G_A = self.G_A.module if self.config['DATA_PARALLEL'] else self.G_A
        G_B = self.G_B.module if self.config['DATA_PARALLEL'] else self.G_B

        self.clamping_alpha(G_A)
        self.clamping_alpha(G_B)

        '''
        ### 1st stage
        # cov_reg: G,C//G,C//G
        # W_reg: B*G,C//G,C//G
        '''
        # get content
        c_A = G_A.c_encoder(x_A)
        c_B = G_B.c_encoder(x_B)

        # get style
        s_A = G_A.s_encoder(x_A)
        s_B = G_B.s_encoder(x_B)

        # from A to B
        x_AB, whitening_reg_AB, coloring_reg_AB = G_B(c_A, s_B) 

        # from B to A
        x_BA, whitening_reg_BA, coloring_reg_BA = G_A(c_B, s_A)

        if isTrain:

            '''
            ### 2nd stage
            '''
            c_BA = G_A.c_encoder(x_BA)
            c_AB = G_B.c_encoder(x_AB)

            s_AB = G_B.s_encoder(x_AB)
            s_BA = G_A.s_encoder(x_BA)

            # from AB to A
            x_ABA, whitening_reg_ABA, coloring_reg_ABA = G_A(c_AB, s_BA)

            # from BA to B
            x_BAB, whitening_reg_BAB, coloring_reg_BAB = G_B(c_BA, s_AB)

            # from A to A
            x_AA, _, _ = G_A(c_A, s_A)
            
            # from B to B
            x_BB, _, _ = G_B(c_B, s_B)

            # Compute the losses
            g_loss_fake = self.D_A.calc_gen_loss(x_BA) + self.D_B.calc_gen_loss(x_AB)

            loss_cross_rec = self.l1_criterion(x_ABA, x_A) + self.l1_criterion(x_BAB, x_B)
            loss_ae_rec = self.l1_criterion(x_AA, x_A) + self.l1_criterion(x_BB, x_B)

            loss_cross_s = self.l1_criterion(s_AB, s_B) + self.l1_criterion(s_BA, s_A)
            
            loss_cross_c = self.l1_criterion(c_AB, c_A) + self.l1_criterion(c_BA, c_B)

            loss_whitening_reg = self.reg([whitening_reg_AB, whitening_reg_BA, whitening_reg_ABA, whitening_reg_BAB])
            loss_coloring_reg = self.reg([coloring_reg_AB, coloring_reg_BA, coloring_reg_ABA, coloring_reg_BAB])

            # Backward and optimize.
            g_loss = g_loss_fake + \
                     self.config['LAMBDA_X_REC'] * (loss_ae_rec) + \
                     self.config['LAMBDA_X_CYC'] * loss_cross_rec + \
                     self.config['LAMBDA_S'] * loss_cross_s + \
                     self.config['LAMBDA_C'] * loss_cross_c + \
                     self.config['LAMBDA_W_REG'] * loss_whitening_reg + \
                     self.config['LAMBDA_C_REG'] * loss_coloring_reg
            
            self.G_optimizer.zero_grad()
            g_loss.backward()
            self.G_optimizer.step()

            # Logging.
            self.loss['G/loss_fake'] = g_loss_fake.item()
            self.loss['G/loss_cross_rec'] = self.config['LAMBDA_X_REC']* loss_cross_rec.item()
            self.loss['G/loss_ae_rec'] = self.config['LAMBDA_X_REC'] * loss_ae_rec.item()
            self.loss['G/loss_latent_c'] = self.config['LAMBDA_C'] * loss_cross_c.item()
            self.loss['G/loss_latent_s'] = self.config['LAMBDA_S'] * loss_cross_s.item()
            self.loss['G/loss_whitening_reg'] = self.config['LAMBDA_W_REG'] * loss_whitening_reg.item()
            self.loss['G/loss_coloring_reg'] = self.config['LAMBDA_C_REG'] * loss_coloring_reg.item()

        return (x_AB, x_BA)

    def update_D(self, x_A, x_B):

        c_A = self.G_A.c_encoder(x_A)
        c_B = self.G_B.c_encoder(x_B)

        s_A = self.G_A.s_encoder(x_A)
        s_B = self.G_B.s_encoder(x_B)

        x_AB, _, _ = self.G_B(c_A, s_B)
        x_BA, _, _ = self.G_A(c_B, s_A)

        # D loss
        d_loss_a = self.D_A.calc_dis_loss(x_BA.detach(), x_A)
        d_loss_b = self.D_B.calc_dis_loss(x_AB.detach(), x_B)
        
        d_loss = d_loss_a + d_loss_b

        self.D_optimizer.zero_grad()
        d_loss.backward()
        self.D_optimizer.step()

        self.loss['D/loss'] = d_loss.item()

    def train(self):

        data_loader = self.data_loader

        print('# iters: %d' % (len(data_loader)))
        print('# data: %d' % (len(data_loader)*self.config['BATCH_SIZE']))
        # Fetch fixed inputs for debugging.
        data_iter = iter(data_loader)

        self.train_ready()
        print("Start training ~ Ayo:)!")
        start_time = time.time()

        
        for i in range(self.config['START'], self.config['NUM_ITERS']):

        ### Preprocess input data ###
            # Fetch real images and labels.
            try:
                x_A, x_B = next(data_iter)
                if x_A.size(0) != self.config['BATCH_SIZE'] or x_B.size(0) != self.config['BATCH_SIZE']:
                    x_A, x_B = next(data_iter)
            except:
                data_iter = iter(data_loader)
                x_A, x_B = next(data_iter)
                if x_A.size(0) != self.config['BATCH_SIZE'] or x_B.size(0) != self.config['BATCH_SIZE']:
                    x_A, x_B = next(data_iter)
            
            x_A = x_A.to(self.device)   # Input images.
            x_B = x_B.to(self.device)   # Exemplar images corresponding with target labels.

        ### Training ###    
            self.update_D(x_A, x_B)
            (x_AB, x_BA)= \
            self.update_G(x_A, x_B)

        ### ETC ###
            if (i+1) % self.config['SAVE_EVERY'] == 0:

                elapsed = time.time() - start_time
                elapsed = str(datetime.timedelta(seconds=elapsed))

                print('=====================================================')
                print("Elapsed [{}], Iter [{}/{}]".format(
                    elapsed, (i+1), self.config['NUM_ITERS']))
                print('=====================================================')
                print('D/loss: %.5f' % (self.loss['D/loss']))
                print('G/loss_fake: %.5f' % (self.loss['G/loss_fake']))
                print('G/loss_cross_rec: %.5f' % (self.loss['G/loss_cross_rec']))
                print('G/loss_ae_rec: %.5f' % (self.loss['G/loss_ae_rec']))
                print('G/loss_latent_s: %.5f' % (self.loss['G/loss_latent_s']))
                print('G/loss_latent_c: %.5f' % (self.loss['G/loss_latent_c']))
                print('G/loss_whitening_reg: %.5f' % (self.loss['G/loss_whitening_reg']))
                print('G/loss_coloring_reg: %.5f' % (self.loss['G/loss_coloring_reg']))
                
                save_img([x_A, x_AB, x_B, x_BA], self.config['SAVE_NAME'], i+1, 'train_results')
                self.model_save(i+1)

            if (i+1) >= self.config['NUM_ITERS_DECAY']:
                self.update_learning_rate()


    def test(self):
        print("test start")
        self.test_ready()

        data_loader = self.data_loader

        with torch.no_grad():
            for i, (x_A, x_B) in enumerate(data_loader):

                x_A = x_A.to(self.device)
                x_B = x_B.to(self.device)

                x_AB, x_BA = self.update_G(x_A, x_B, isTrain=False)
                save_img([x_A, x_B, x_AB, x_BA], self.config['SAVE_NAME'], i+1, 'test_results')

    def test_new(self, mode):
        print('New test start')
        self.test_ready()

        blank_img = Image.open(self.config['FOR_TEST']['blank_img'])
        blank_img = blank_img.resize((self.config['RESIZE'], self.config['RESIZE']))
        to_tensor = T.ToTensor()
        blank_img = to_tensor(blank_img).unsqueeze(0)

        data_loader = self.data_loader

        for i, (x_A, x_B) in enumerate(data_loader):

            x_A = x_A.to(self.device)
            x_B = x_B.to(self.device)
            break
        
        if mode == 'A2B':
            content = x_A
            style = x_B
        elif mode == 'B2A':
            content = x_B
            style = x_A
        else:
            assert 0, 'Unknown mode'

        imgs = [blank_img]
        for i in range(content.size(0)):
            img = (content[i].unsqueeze(0).cpu().data+1)/2
            imgs.append(img)

        for j in range(style.size(0)):
            s = style[j].unsqueeze(0)
            img = (s.cpu().data+1)/2
            imgs.append(img)
            for i in range(content.size(0)):
                c = content[i].unsqueeze(0)
                if mode == 'A2B':
                    img, _ = self.update_G(c, s, isTrain=False)
                else:
                    _, img = self.update_G(s, c, isTrain=False)
                img = (img.cpu().data+1)/2
                imgs.append(img)
        self.save_img_new(self.config['FOR_TEST']['save_dir'], mode+'.jpg', imgs, style.size(0)+1)

    def debug(self):
        print('Debug start (visualize content and style)')
        self.test_ready()

        data_loader = self.data_loader

        for i, (x_A, x_B) in enumerate(data_loader):

            x_A = x_A.to(self.device)
            x_B = x_B.to(self.device)
            break
        
        c_A, s_A = self.get_latent(x_A, 'A')
        c_B, s_B = self.get_latent(x_B, 'B')

        self.vis_mask(x_A, c_A, s_A, 'A')
        self.vis_mask(x_B, c_B, s_B, 'B')

    def vis_mask(self, imgs, content, style, domain='A'):
        plt.figure(figsize=(12,6))
        num_imgs = imgs.size(0)
        num_plot = 5
        index_c = np.random.permutation(content.size(1))
        index_s = np.random.permutation(style.size(1))

        for i in range(num_imgs):
            img = imgs[i].cpu().numpy()
            c = content[i].cpu().numpy()
            s = style[i].cpu().numpy()
            img = np.transpose(img, (1,2,0))
            
            sub_id = i*(num_plot*2+1)+1
            plt.subplot(num_imgs, num_plot*2+1, sub_id)
            plt.imshow(img)
            plt.axis('off')
            plt.title('input')
            
            index_c = np.random.permutation(index_c)
            for j in range(num_plot):
                k = index_c[j]
                plt.subplot(num_imgs, num_plot*2+1, sub_id+j+1)
                tc = c[k, :, :]
                min_c = np.min(tc)
                max_c = np.max(tc)
                tc = (tc-min_c) / (max_c-min_c+1e-5)
                plt.imshow(tc)
                plt.axis('off')
                plt.title('c-%d'%k)

            index_s = np.random.permutation(index_s)
            for j in range(num_plot):
                k = index_s[j]
                plt.subplot(num_imgs, num_plot*2+1, sub_id+num_plot+j+1)
                ts = s[k, :, :]
                min_s = np.min(ts)
                max_s = np.max(ts)
                ts = (ts-min_s) / (max_s-min_s+1e-5)
                plt.imshow(ts)
                plt.axis('off')
                plt.title('s-%d'%k)

        plt.savefig(os.path.join(self.config['FOR_DEBUG']['save_dir'], mode+'_latent.jpg'))

    def save_img_new(self, save_dir, save_name, images, num_input):
        image_tensor = torch.cat(images, 0)
        image_grid = vutils.make_grid(image_tensor.data, nrow=num_input, padding=0, normalize=True)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        vutils.save_image(image_grid, os.path.join(save_dir, save_name), nrow=1)
        

def main(config):
    
    # For fast training
    cudnn.benchmark = True

    mode = config['MODE']
    if mode == 'train':
        run = Run(config)
        run.train()
    elif mode == 'debug':
        config['MODE'] = config['mode4loader']
        run = Run(config)
        run.debug()
    else:
        config['MODE'] = config['mode4loader']
        run = Run(config)
        run.test()
        # run.test_new('A2B')
        # run.test_new('B2A')

config = ges_Aonfig(sys.argv[1])

main(config)
