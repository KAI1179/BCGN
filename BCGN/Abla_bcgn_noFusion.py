from lavis.models import load_model_and_preprocess
from ReasonGrasp.layers import TransformerDecoder
import torchvision.models as Models
import torch.nn.functional as F
import torch
import torch.nn as nn


# model, vis_processors, txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)

class grasp_net(nn.Module):
    def __init__(self, input_size=224, dropout=False, prob=0.1, device='cpu'):
        super(grasp_net, self).__init__()

        self.device = device
        # self.blip, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True, device=device)
        self.blip, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=True,  device=device)  ## channel==768
        # self.blip, self.vis_processors, self.txt_processors = load_model_and_preprocess(name="blip_feature_extractor", model_type="base", is_eval=False,  device=device)  ## channel==768
        # self.blip_feature = self.blip.extract_features

        for param in self.blip.parameters():
            param.requires_grad = False

        self.resnet50 = Models.resnet50(pretrained=True)

        self.vis_trans = nn.Sequential(
            nn.Conv2d(1024, 768, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(768),
            nn.ReLU(),
        )

        # self.match = nn.Sequential()  ## IMG_input = B * 197 * 768  TXT_input = B * 5 * 768   ## output = B * 768 * 14 * 14
        self.match = TransformerDecoder(num_layers=1,   # cfg.num_layers,  ## 1
                                        d_model=768,  # cfg.vis_dim,
                                        nhead=8,      # cfg.num_head,
                                        dim_ffn=2048, # cfg.dim_ffn,
                                        dropout=0.1,  # cfg.dropout,
                                        return_intermediate=False) # cfg.intermediate)
        # self.change_match = nn.Conv1d()

        self.tran_conv = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=(3, 3), stride=(4, 4), padding=(0, 0), output_padding=(1, 1)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), output_padding=(1, 1)),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), output_padding=(1, 1)),
        )

        self.txt_global_up = nn.Sequential(
            nn.Conv1d(768, 32, kernel_size=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )


        self.pos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.cos_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.sin_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)
        self.width_output = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=(3, 3), padding=1)

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)

        self.judge_conv = nn.Sequential(
            nn.Conv2d(768, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=(3, 3)),
            nn.BatchNorm2d(512),
        )
        self.judge_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.judge_classifier = nn.Linear(512, 2)




    def forward(self, x_img, x_txt):
        # pad_mask input: {b, L}
        _x_text = []
        for i in x_txt:
            _x_text.append(self.txt_processors["eval"](i))

        x = {"image": x_img, "text_input": _x_text}


        # fea_blip_img = self.blip.extract_features(x, mode="image").image_embeds      # B, 197, 768
        # fea_blip_txt = self.blip.extract_features(x, mode="text").text_embeds   # B, 5, 768
        fea_blip_img = self.blip.extract_features(x, mode="image")
        fea_blip_img_high = fea_blip_img.image_embeds  # B, 197, 768
        # fea_blip_img_low = fea_blip_img.image_embeds_proj  # B, 197, 256

        fea_blip_img_local = fea_blip_img_high[:, 1:, :].permute(0, 2, 1)  # B, 768, 14*14
        fea_blip_img_local = fea_blip_img_local.reshape(-1, 768, 14, 14)  # B, 768, 14, 14
        # fea_blip_img_global = fea_blip_img_high[:, 0, :]  # B, 768


        fea_blip_txt = self.blip.extract_features(x, mode="text")
        fea_blip_txt_high = fea_blip_txt.text_embeds  # B, 197, 768
        # fea_blip_txt_low = fea_blip_txt.text_embeds_proj  # B, 197, 256
        fea_blip_txt_local = fea_blip_txt_high[:, 1:, :]  # B, L, 768
        fea_blip_txt_global = fea_blip_txt_high[:, 0, :].unsqueeze(2)  # B, 768, 1

        # pad_mask = torch.zeros(fea_blip_txt.size()[0:2], device=fea_blip_txt.device).masked_fill_(fea_blip_txt[-1].mean() == 0, 1).bool()
        # pad_mask = torch.zeros(fea_blip_txt.size()[0:2], device=self.device).masked_fill_(fea_blip_txt[-1].mean() == 0, 1).bool()
        pad_mask = torch.zeros(fea_blip_txt_local.size()[0:2], device=self.device).masked_fill_(fea_blip_txt_local[-1].mean() == 0, 1).bool()  # ## 这里应该无用，用于transformer decoder时
        # pad_mask.to(self.device)

        fea_res_img = self.resnet50.conv1(x_img)
        fea_res_img = self.resnet50.bn1(fea_res_img)
        fea_res_img = self.resnet50.relu(fea_res_img)
        fea_res_img = self.resnet50.maxpool(fea_res_img)  ## 64 * 56 * 56
        fea_res_img = self.resnet50.layer1(fea_res_img)  ## 256 * 56 * 56
        fea_res_img = self.resnet50.layer2(fea_res_img)  ## 512 * 28 * 28
        fea_res_img = self.resnet50.layer3(fea_res_img)  ## 1024 * 14 * 14
        # fea_res_img = self.resnet50.layer4(fea_res_img)  ## 2048 * 7 * 7

        fea_res_img = self.vis_trans(fea_res_img)  ## b, 768, 14, 14  ## img特征的维度变换


        # match input:  {vis: b, 512, h, w};  {txt: b, L, 512};   {pad_mask: b, L}
        # match output: {matched_feat: b, 512, HW}
        # matched_fea = self.match(fea_blip_img_local, fea_blip_txt_local, pad_mask)  # ## B 768 14*14 ## 跨模态融合
        # matched_fea = matched_fea.reshape(-1, 768, 14, 14)

        matched_fea = fea_blip_img_local




        final_fea = fea_res_img + matched_fea  ## 跨模态特征 + 图像特征


        # final_fea = self.tran_conv(matched_fea)  # ##  B 32 224 224
        final_fea = self.tran_conv(final_fea)  # ##  B 32 224 224

        fea_blip_txt_global_up = self.txt_global_up(fea_blip_txt_global)
        fea_blip_txt_global_up = fea_blip_txt_global_up.unsqueeze(3).contiguous()

        # final_fea_pos = final_fea * fea_blip_txt_global_up
        final_fea = final_fea * fea_blip_txt_global_up


        if self.dropout:
            # pos_output = self.pos_output(self.dropout_pos(final_fea))
            pos_output = self.pos_output(self.dropout_pos(final_fea))
            cos_output = self.cos_output(self.dropout_cos(final_fea))
            sin_output = self.sin_output(self.dropout_sin(final_fea))
            width_output = self.width_output(self.dropout_wid(final_fea))
        else:
            # pos_output = self.pos_output(final_fea)
            pos_output = self.pos_output(final_fea)
            cos_output = self.cos_output(final_fea)
            sin_output = self.sin_output(final_fea)
            width_output = self.width_output(final_fea)

        ## 判断是否存在指令限定的物体
        fea_blip_txt_global_judge = fea_blip_txt_global.unsqueeze(3).contiguous()
        judge_feature = matched_fea * fea_blip_txt_global_judge ## (B, 768, 14, 14) * (B, 768, 1)
        judge_feature = self.judge_conv(judge_feature)
        judge_feature = self.judge_pool(judge_feature)
        judge_feature = judge_feature.view(judge_feature.size(0), -1)
        is_available = self.judge_classifier(judge_feature)



        return pos_output, cos_output, sin_output, width_output, is_available


if __name__ == '__main__':
    # device = "cuda:3" if torch.cuda.is_available() else "cpu"
    model = grasp_net()
    print(model)
    x = torch.randn([2, 3, 224, 224])
    text = ["spewing spewing spewing.", "spewing spewing spewing."]

    y, _, _, _, is_avai = model(x, text)

    # y = model(x)
    print(y.shape)
    print(is_avai)
    print('...')


