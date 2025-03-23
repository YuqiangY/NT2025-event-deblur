import torch


def tta(LRs_RGB, events, net, mode='v'):
    
    HRs_RGB = []
    with torch.no_grad():
        # 原图
        HRs_RGB_org = net(LRs_RGB, events)[-1]
        HRs_RGB.append(HRs_RGB_org)

        # 垂直翻转
        if 'fv+' in mode:
            print('fv+')
            LRs_RGB_1 = torch.flip(LRs_RGB, dims=[-2])
            events_1 = torch.flip(events, dims=[-2])
            HRs_RGB_1 = net(LRs_RGB_1, events_1)[-1]
            HRs_RGB.append(torch.flip(HRs_RGB_1, dims=[-2]))
    
        # 水平翻转
        if 'fh+' in mode:
            print('fh+')
            LRs_RGB_1 = torch.flip(LRs_RGB, dims=[-1])
            events_1 = torch.flip(events, dims=[-1])
            HRs_RGB_1 = net(LRs_RGB_1, events_1)[-1]
            HRs_RGB.append(torch.flip(HRs_RGB_1, dims=[-1]))
        
        # 水平垂直翻转
        if 'fvh+' in mode:
            print('fvh+')
            LRs_RGB_1 = torch.flip(LRs_RGB, dims=[-2, -1])
            events_1 = torch.flip(events, dims=[-2, -1])
            HRs_RGB_1 = net(LRs_RGB_1, events_1)[-1]
            HRs_RGB.append(torch.flip(HRs_RGB_1, dims=[-2, -1]))

        # # RGB通道打乱
        # if 'rgb+' in mode:
        #     print('rgb+')
        #     LRs_RGB_1 = LRs_RGB[:, [0, 2, 1], :, :] # rbg
        #     HRs_RGB_1 = net(LRs_RGB_1, events)[-1]
        #     HRs_RGB.append(HRs_RGB_1[:, [0, 2, 1], :, :]) # rgb

        #     LRs_RGB_1 = LRs_RGB[:, [1, 0, 2], :, :] # grb
        #     HRs_RGB_1 = net(LRs_RGB_1, events)[-1]
        #     HRs_RGB.append(HRs_RGB_1[:, [1, 0, 2], :, :]) # rgb

        #     LRs_RGB_1 = LRs_RGB[:, [1, 2, 0], :, :] # gbr
        #     HRs_RGB_1 = net(LRs_RGB_1, events)[-1]
        #     HRs_RGB.append(HRs_RGB_1[:, [2, 0, 1], :, :]) # rgb

        #     LRs_RGB_1 = LRs_RGB[:, [2, 0, 1], :, :] # brg
        #     HRs_RGB_1 = net(LRs_RGB_1, events)[-1]
        #     HRs_RGB.append(HRs_RGB_1[:, [1, 2, 0], :, :]) # rgb

        #     LRs_RGB_1 = LRs_RGB[:, [2, 1, 0], :, :] # bgr
        #     HRs_RGB_1 = net(LRs_RGB_1, events)[-1]
        #     HRs_RGB.append(HRs_RGB_1[:, [2, 1, 0], :, :]) # rgb

        # rot90
        if 'rot90+' in mode:
            print('rot90+')
            LRs_RGB_1 = torch.rot90(LRs_RGB, dims=[-2, -1])
            events_1 = torch.rot90(events, dims=[-2, -1])
            HRs_RGB_1 = net(LRs_RGB_1, events_1)[-1]
            HRs_RGB.append(torch.rot90(HRs_RGB_1, k=-1, dims=[-2, -1]))

    HRs_RGB = torch.stack(HRs_RGB, dim=0)
    HRs_RGB = torch.mean(HRs_RGB, dim=0)

    return HRs_RGB