from lavis.models import load_model_and_preprocess
from inference.models.layers import TransformerDecoder
from inference.models.grasp_model import GraspModel, ResidualBlock

import torch
import torch.nn as nn
import torch.nn.functional as F

# device = "cuda:3" if torch.cuda.is_available() else "cpu"

# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

class grasp_net(GraspModel):
    def __init__(self, input_size=224, dropout=False, prob=0.1, device='cpu'):
        super(grasp_net, self).__init__()

        self.device = device
        # self.blip, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
        # self.blip, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True,  device=device)  ## channel==768
        # self.blip, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True,  device=device)  ## channel==768
        self.blip, _, _ = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=False,  device=device)  ## channel==768
        # self.blip_feature = self.blip.extract_features

        # self.match = nn.Sequential()  ## IMG_input = B * 197 * 768  TXT_input = B * 5 * 768   ## output = B * 768 * 14 * 14
        self.match = TransformerDecoder(num_layers=3,   # cfg.num_layers,
                                        d_model=768,  # cfg.vis_dim,
                                        nhead=8,      # cfg.num_head,
                                        dim_ffn=2048, # cfg.dim_ffn,
                                        dropout=0.1,  # cfg.dropout,
                                        return_intermediate=False) # cfg.intermediate)
        ## grasp_head ↓
        # channel_size = 32
        # output_channels = 1
        # prob = 0.1
        # dropout = False

        # self.conv_che = nn.Sequential(
        #     nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(1, 1)),  ## 降维度
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(),
        # )

        self.conv_instand_match = nn.Sequential(
            nn.Conv2d(in_channels=768, out_channels=512, kernel_size=(2, 2), padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2, 2)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )


        self.tran_conv = nn.Sequential(
            # nn.ConvTranspose2d(768, 256, kernel_size=(3, 3), stride=(4, 4), padding=(0, 0), output_padding=(1, 1)),
            nn.ConvTranspose2d(512, 256, kernel_size=(3, 3), stride=(4, 4), padding=(0, 0), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1)),
        )

        # self.conv4 = nn.ConvTranspose2d(768, 256, kernel_size=(3, 3), stride=(4, 4), padding=(0, 0), output_padding=(1, 1))
        # self.bn4 = nn.BatchNorm2d(256)
        # self.conv5 = nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1))
        # self.bn5 = nn.BatchNorm2d(64)
        # self.conv6 = nn.ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1))



        self.pos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.cos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.sin_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.width_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        # for m in self.tran_conv():
        #     if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        #         nn.init.xavier_uniform_(m.weight, gain=1)

        ## grasp_head ↑

    def forward(self, x_img):
        # x_txt = []
        # for i in range(x_img.shape[0]):
        #     x_txt.append('Grasp that object')
        #
        # # pad_mask input: {b, L}
        # # _x_text = []
        # # for i in x_txt:
        # #     _x_text.append(self.txt_processors["eval"](i))
        #
        x = {"image": x_img, "text_input": None}


        fea_blip_img = self.blip.extract_features(x, mode="image").image_embeds      # B, 197, 768
        # fea_blip_txt = self.blip.extract_features(x, mode="text").text_embeds   # B, 5, 768

        # pad_mask = torch.zeros(fea_blip_txt.size()[0:2], device=fea_blip_txt.device).masked_fill_(fea_blip_txt[-1].mean() == 0, 1).bool()
        # pad_mask = torch.zeros(fea_blip_txt.size()[0:2], device=self.device).masked_fill_(fea_blip_txt[-1].mean() == 0, 1).bool()
        # pad_mask.to(self.device)

        fea_blip_img = fea_blip_img[:, 1:, :].permute(0, 2, 1)
        fea_blip_img = fea_blip_img.reshape(-1, 768, 14, 14)

        # match input:  {vis: b, 512, h, w};  {txt: b, L, 512};   {pad_mask: b, L}
        # match output: {matched_feat: b, 512, HW}
        # matched_fea = self.match(fea_blip_img, fea_blip_txt, pad_mask)
        # matched_fea = matched_fea.reshape(-1, 768, 14, 14)

        # matched_fea = self.conv4(matched_fea)
        # matched_fea = self.conv5(matched_fea)
        # matched_fea_1 = self.conv6(matched_fea)  ##
        # matched_fea_2 = self.conv7(matched_fea)  ##

        # final_fea = self.conv_che(fea_blip_img)
        final_fea = self.conv_instand_match(fea_blip_img)


        final_fea = self.tran_conv(final_fea)



        if self.dropout:
            pos_output = self.pos_output(self.dropout_pos(final_fea))
            cos_output = self.cos_output(self.dropout_cos(final_fea))
            sin_output = self.sin_output(self.dropout_sin(final_fea))
            width_output = self.width_output(self.dropout_wid(final_fea))
        else:
            pos_output = self.pos_output(final_fea)
            cos_output = self.cos_output(final_fea)
            sin_output = self.sin_output(final_fea)
            width_output = self.width_output(final_fea)

        return pos_output, cos_output, sin_output, width_output



if __name__ == '__main__':
    model = grasp_net()
    print(model)
    x = torch.randn([1, 3, 224, 224])
    # x = torch.randn([1, 3, 300, 300])
    text = "spewing"
    # x_1, x_2, x_3_1, x_3_2, x_3_3, x_4 = model(x)
    # print(x_1.shape, x_2.shape, x_3_1.shape, x_3_2.shape, x_3_3.shape, x_4.shape)

    y, _, _, _ = model(x)

    # y = model(x)
    print(y.shape)
    # g = make_dot(y)
    # g.render('models_small_full', view=False)

