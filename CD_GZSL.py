from __future__ import print_function
import argparse
import sys

sys.path.append("..")
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import util
import classifier_embed_contras_S
import model_CD
import losses
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr
parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--matdataset', default=True, help='Data in matlab format')
parser.add_argument('--image_embedding', default='res101')

parser.add_argument('--syn_num', type=int, default=300, help='number features to generate per class')
parser.add_argument('--gzsl', type=bool, default=True, help='enable generalized zero-shot learning')
parser.add_argument('--preprocessing', type=bool, default=True, help='enbale MinMaxScaler on visual features')
parser.add_argument('--standardization', action='store_true', default=False)
parser.add_argument('--validation', action='store_true', default=False, help='enable cross validation mode')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batch_size', type=int, default=1024, help='input batch size')
parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
# parser.add_argument('--attSize', type=int, default=102, help='size of semantic features')
parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
parser.add_argument('--nz', type=int, default=1024, help='noise for generation')
parser.add_argument('--embedSize', type=int, default=2048, help='size of embedding h')
parser.add_argument('--outzSize', type=int, default=1024, help='size of non-liner projection z')

## network architechure
parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator G')
parser.add_argument('--ndh', type=int, default=4096, help='size of the hidden units in discriminator D')
parser.add_argument('--nhF', type=int, default=2048, help='size of the hidden units comparator network F')

parser.add_argument('--ins_weight', type=float, default=0.001, help='weight of the classification loss when learning G')
parser.add_argument('--cls_weight', type=float, default=0.001, help='weight of the score function when learning G')
parser.add_argument('--ins_temp', type=float, default=0.1, help='temperature in instance-level supervision')
parser.add_argument('--cls_temp', type=float, default=0.1, help='temperature in class-level supervision')
parser.add_argument('--au_temp', type=float, default=10, help='temperature in class-level supervision')
parser.add_argument('--fake_temp', type=float, default=1.0, help='temperature in class-level supervision')

parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate to training')
parser.add_argument('--lr_decay_epoch', type=int, default=10, help='conduct learning rate decay after every 100 epochs')
parser.add_argument('--lr_dec_rate', type=float, default=0.99, help='learning rate decay rate')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
parser.add_argument('--manualSeed', type=int, default=3483, help='manual seed')
parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of all classes')
parser.add_argument('--hidden_dim', type=int, default=1024, help='size of non-liner projection z')
parser.add_argument('--gpus', default='1', help='the number of the GPU to use')
# z_dim = hs_dim + hu_dim 这是VAE的隐变量 假设相同纬度 如果不相同，再进行讨论
parser.add_argument('--hs_dim', type=int, default='256', help='dimension of latent variable of sematic-relation feature')
parser.add_argument('--hu_dim', type=int, default='256',
                    help='dimension of latent variable of non-sematic-relation feature')
parser.add_argument('--dataset', default='CUB', help=' ')
parser.add_argument('--class_embedding', default='sent', help='att or sent')
# parser.add_argument('--class_embedding', default='att',help='att or sent')
parser.add_argument('--dir', default='none',help='dir of   model')
opt = parser.parse_args()
if opt.dataset == 'CUB':
    opt.resSize = 2048
    opt.nclass_all = 200
    opt.dir = 'models\CUB'
    opt.nclass_seen = 150
    opt.fake_temp = 0.01
    if opt.class_embedding == 'att':
        opt.batch_size = 128
        opt.nz = 312
        opt.attSize = 312
        opt.hs_dim = 64
        opt.hu_dim = 64
    else:# sent
        opt.batch_size = 1024
        opt.nz = 1024
        opt.attSize = 1024
        opt.lr = 1e-4
        opt.syn_num = 100
        opt.manualSeed = 1234
        opt.au_temp = 1e-3
        opt.fake_temp = 0.01
        opt.ins_temp = 0.1
        opt.cls_temp = 0.1
        dong = True
        opt.hs_dim = 2048
        opt.hu_dim = 2048
        opt.outzSize = 1024

print(opt)

# os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# print(torch.cuda.get_device_name(0),torch.cuda.get_device_name(1))
torch.cuda.set_device(0)
if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

# load data
data = util.DATA_LOADER(opt)
print("# of training samples: ", data.ntrain)

# CGAN的生成模块
netG = model_CDR.MLP_G(opt)
netD = model_CDR.MLP_CRITIC(opt)
# AE部分
netEncoder = model_CDR.Encoder(opt)
netDecoder = model_CDR.Decoder(opt)
# 相当于 Relation model 目的是计算相容性分数 (hu信息是否可当做负样本对？
F_ha = model_CDR.Dis_Embed_Att(opt)
D_TC = model_CDR.Dis_TC(opt)

model_path = './models/' + opt.dataset
if not os.path.exists(model_path):
    os.makedirs(model_path)

if len(opt.gpus.split(',')) > 1:
    netG = nn.DataParallel(netG)
    netD = nn.DataParallel(netD)
    netEncoder = nn.DataParallel(netEncoder)
    netDecoder = nn.DataParallel(netDecoder)
    F_ha = nn.DataParallel(F_ha)

# latent space的L2_norm的对比学习模块
contras_criterion = losses.SupConLoss_clear(opt.ins_temp)

input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
# input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
noise_gen = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)

if opt.cuda:
    netG.cuda()
    netD.cuda()
    netEncoder.cuda()
    netDecoder.cuda()
    F_ha.cuda()
    D_TC.cuda()
    input_res = input_res.cuda()
    noise_gen, input_att = noise_gen.cuda(), input_att.cuda()
    input_label = input_label.cuda()


def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))


def generate_syn_feature(netG, classes, attribute, num):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass * num, opt.resSize)
    syn_label = torch.LongTensor(nclass * num)
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()

    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            output = netG(syn_noise, syn_att)
        syn_feature.narrow(0, i * num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i * num, num).fill_(iclass)
    return syn_feature, syn_label


# setup optimizer
import itertools

optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerE = optim.Adam(netEncoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerF = optim.Adam(F_ha.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerDecoder = optim.Adam(netDecoder.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerTC = optim.Adam(D_TC.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

schedulerD = lr.CosineAnnealingLR(optimizerD,T_max=5)
schedulerG = lr.StepLR(optimizerG,step_size=170,gamma=0.2)
schedulerE = lr.StepLR(optimizerE,step_size=170,gamma=0.2)
schedulerF = lr.CosineAnnealingLR(optimizerF,T_max=5)
schedulerDecoder = lr.CosineAnnealingLR(optimizerDecoder,T_max=5)
schedulerTC = lr.CosineAnnealingLR(optimizerTC,T_max=5)


def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    # print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, input_att)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


# use the for-loop to save the GPU-memory
def class_scores_for_loop(embed, input_label, relation_net):
    all_scores = torch.FloatTensor(embed.shape[0], opt.nclass_seen).cuda()
    for i, i_embed in enumerate(embed):
        expand_embed = i_embed.repeat(opt.nclass_seen, 1)  # .reshape(embed.shape[0] * opt.nclass_seen, -1)
        all_scores[i] = (torch.div(relation_net(torch.cat((expand_embed, data.attribute_seen.cuda()), dim=1)),
                                   opt.cls_temp).squeeze())
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

def class_scores_for_loop_hs_hu(embed_hs,embed_hu, input_label, relation_net):
    #传入的embed是hs和hu拼接而成的 但是label只有hs的 原因是hu不属于任何一类 只充当分母
    # print(embed_hs.shape)
    all_scores = torch.FloatTensor(embed_hs.shape[0], 2 * opt.nclass_seen).cuda()
    sematic = data.attribute_seen
    sematic = torch.cat((sematic, sematic), dim=0).cuda()
    for i, i_embed in enumerate(embed_hs):
        expand_embed_hs = i_embed.repeat(opt.nclass_seen, 1)  # .reshape(embed.shape[0] * opt.nclass_seen, -1)
        expand_embed_hu = embed_hu[i].repeat(opt.nclass_seen, 1)  # .reshape(embed.shape[0] * opt.nclass_seen, -1)
        expand_embed = torch.cat((expand_embed_hs,expand_embed_hu),dim=0)
        all_scores[i] = (torch.div(relation_net(torch.cat((expand_embed,sematic), dim=1)),
                                   opt.cls_temp).squeeze())

    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # normalize the scores for stable training
    scores_norm = all_scores - score_max.detach()
    # 这个mask只是hs的
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    # 再给hu拼接一个label全为0的mask
    mask_hu = torch.zeros_like(mask).cuda()
    mask = torch.cat((mask,mask_hu),dim=1)
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    # print(mask.shape,log_scores.shape)
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss


# It is much faster to use the matrix, but it cost much GPU memory.
def class_scores_in_matrix(embed, input_label, relation_net):
    expand_embed = embed.unsqueeze(dim=1).repeat(1, opt.nclass_seen, 1).reshape(embed.shape[0] * opt.nclass_seen, -1)
    expand_att = data.attribute_seen.unsqueeze(dim=0).repeat(embed.shape[0], 1, 1).reshape(
        embed.shape[0] * opt.nclass_seen, -1).cuda()
    all_scores = torch.div(relation_net(torch.cat((expand_embed, expand_att), dim=1)), opt.cls_temp).reshape(
        embed.shape[0],
        opt.nclass_seen)
    score_max, _ = torch.max(all_scores, dim=1, keepdim=True)
    # ?
    scores_norm = all_scores - score_max.detach()
    mask = F.one_hot(input_label, num_classes=opt.nclass_seen).float().cuda()
    exp_scores = torch.exp(scores_norm)
    log_scores = scores_norm - torch.log(exp_scores.sum(1, keepdim=True))
    cls_loss = -((mask * log_scores).sum(1) / mask.sum(1)).mean()
    return cls_loss

l1_loss = nn.L1Loss(size_average=False)
for epoch in range(opt.nepoch):
    FP = 0
    mean_lossD = 0
    mean_lossG = 0

    for i in range(0, data.ntrain, opt.batch_size):
        ###########################

        # 对于VAE
        for _ in range(1):
            for p in netEncoder.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in netDecoder.parameters():  # reset requires_grad
                p.requires_grad = True
            for p in F_ha.parameters():  # reset requires_grad
                p.requires_grad = True
            # TC model
            for _ in range(1):
                D_TC.zero_grad()
                sample()
                _, hs_real, hu_real, z_real, hs_l2_real = netEncoder(input_res)
                ture_value_ = D_TC(hs_real,hu_real)
                ture_value = torch.mean(ture_value_)
                fake_value_ = D_TC(hs_real,util.shuffle_z(hu_real))
                fake_value = torch.mean(fake_value_)
                loss_tc = fake_value - ture_value
                loss_tc.backward()
                optimizerTC.step()
            # print(loss_tc)
            for p in D_TC.parameters():  # reset requires_grad
                p.requires_grad = False
            netEncoder.zero_grad()
            netDecoder.zero_grad()
            F_ha.zero_grad()
            sample()
            _, hs_real, hu_real, z_real, hs_l2_real = netEncoder(input_res)
            rec = netDecoder(z_real)
            rec_loss = l1_loss(input_res, rec)
            real_ins_contras_loss = contras_criterion(hs_l2_real, input_label)
            cls_loss_real = class_scores_for_loop_hs_hu(hs_real,hu_real,input_label,F_ha)
            tc_loss = D_TC(hs_real,hu_real)
            tc_loss = torch.mean(tc_loss)
            # KLD = (0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()))
            lossvae = opt.au_temp * rec_loss  + cls_loss_real+ real_ins_contras_loss - tc_loss\


                      # - KLD
            lossvae.backward()
            optimizerE.step()
            optimizerDecoder.step()
            optimizerF.step()
            # print('rec_loss = %f, real_ins_contras_loss = %f,cls_loss_real = %f' %
            #       (rec_loss.item(),real_ins_contras_loss.item(),cls_loss_real.item()))
        for p in netDecoder.parameters():  # reset requires_grad
            p.requires_grad = False
        # train D
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update
        for p in F_ha.parameters():  # reset requires_grad
            p.requires_grad = False

        for iter_d in range(opt.critic_iter):
            # 获取数据
            sample()

            netD.zero_grad()
            netEncoder.zero_grad()

            # 训练判别器部分
            criticD_real = netD(input_res, input_att)
            # print('D', criticD_real)
            criticD_real = criticD_real.mean()

            # train with fakeG
            noise_gen.normal_(0, 1)
            fake = netG(noise_gen, input_att)
            # 切断梯度传向 GAN的生成器部分
            criticD_fake = netD(fake.detach(), input_att)
            criticD_fake = criticD_fake.mean()
            # gradient penalty
            gradient_penalty = calc_gradient_penalty(netD, input_res, fake.data, input_att)
            Wasserstein_D = criticD_real - criticD_fake
            D_cost = criticD_fake - criticD_real + gradient_penalty  \
                     # + cls_loss_real #+  real_ins_contras_loss
            # print('D',criticD_real.item(),criticD_fake.item(),real_ins_contras_loss.item(),cls_loss_real.item())
            D_cost.backward()
            optimizerD.step()
            # optimizerE.step()
        ############################
        # (2) Update G network: optimize WGAN-GP objective, Equation (2)
        ###########################
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = False  # avoid computation
        for p in netEncoder.parameters():  # reset requires_grad
            p.requires_grad = False

        netG.zero_grad()
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen, input_att)


        criticG_fake = netD(fake, input_att)
        criticG_fake = criticG_fake.mean()
        G_cost = -criticG_fake

        # _, hs_real, _, _, hs_l2_real = netEncoder(input_res)


        # print(fake_ins_contras_loss)

        # cls_loss_fake = class_scores_for_loop(hs_fake, input_label, F_ha)

        errG = G_cost \
               # + cls_loss_fake \
               # + opt.ins_weight * fake_ins_contras_loss \
                 # + opt.ins_weight * c_errG
        # print('G',G_cost.item())

        errG.backward()
        optimizerG.step()
        # train G E for the purpose of generate img which help disentangle z to hs and hu better
        # instance-level
        netG.zero_grad()
        netEncoder.zero_grad()
        F_ha.zero_grad()
        _, hs_real, hu_real, _, hs_l2_real = netEncoder(input_res)
        # real_ins_contras_loss = contras_criterion(hs_l2_real, input_label)
        noise_gen.normal_(0, 1)
        fake = netG(noise_gen, input_att)
        _, hs_fake, hu_fake, _, hs_l2_fake = netEncoder(fake)
        all_outz = torch.cat((hs_l2_fake, hs_l2_real.detach()), dim=0)
        fake_ins_contras_loss = contras_criterion(all_outz, torch.cat((input_label, input_label), dim=0))
        # class-level
        cls_loss_fake = class_scores_for_loop_hs_hu(hs_fake,hu_fake, input_label, F_ha)
        loss_four = opt.fake_temp * (fake_ins_contras_loss + cls_loss_fake)#
        loss_four.backward()
        optimizerE.step()
        #备注1下面那两个网络参数不更新时 取得的效果很好
        # optimizerG.step()
        # optimizerF.step()
    if dong:
        schedulerE.step()
        schedulerG.step()

    if (epoch + 1) % opt.lr_decay_epoch == 0:
        for param_group in optimizerD.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate
        for param_group in optimizerG.param_groups:
            param_group['lr'] = param_group['lr'] * opt.lr_dec_rate

    mean_lossG /= data.ntrain / opt.batch_size
    mean_lossD /= data.ntrain / opt.batch_size
    print(
        '[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, real_ins_contras_loss:%.4f, fake_ins_contras_loss:%.4f, cls_loss_real: %.4f, cls_loss_fake: %.4f'  # , cls_loss_real: %.4f, cls_loss_fake: %.4f
        % (
        epoch, opt.nepoch, D_cost, G_cost, Wasserstein_D, real_ins_contras_loss, fake_ins_contras_loss, cls_loss_real,
        cls_loss_fake))  # , cls_loss_real, cls_loss_fake

    # evaluate the model, set G to evaluation mode
    netG.eval()

    for p in netEncoder.parameters():  # reset requires_grad
        p.requires_grad = False
    # if epoch == 26:
    # torch.save(netEncoder.state_dict(),os.path.join(opt.dir,'enc_awa2_{}.pkl'.format(epoch)))
    # torch.save(netG.state_dict(),os.path.join(opt.dir,'gen_awa2_{}.pkl'.format(epoch)))

    if opt.gzsl:  # Generalized zero-shot learning
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        train_X = torch.cat((data.train_feature, syn_feature), 0)
        train_Y = torch.cat((data.train_label, syn_label), 0)

        nclass = opt.nclass_all

        cls = classifier_embed_contras_S.CLASSIFIER(train_X, train_Y, netEncoder, opt.hs_dim, data, nclass, opt.cuda,
                                                  opt.classifier_lr, 0.5, 25, opt.syn_num,
                                                  True)
        print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))

    else:  # conventional zero-shot learning
        syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num)
        cls = classifier_embed_contras_S.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), netEncoder,
                                                  opt.embedSize, data,
                                                  data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, 100,
                                                  opt.syn_num,
                                                  False)
        acc = cls.acc
        print('unseen class accuracy=%.4f ' % acc)

    # reset G to training mode
    netG.train()
    for p in netEncoder.parameters():  # reset requires_grad
        p.requires_grad = True
