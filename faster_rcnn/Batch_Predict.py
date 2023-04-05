import os
import time
import json

import torch
import torchvision
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from network_files import FasterRCNN, FastRCNNPredictor, AnchorsGenerator
from backbone import resnet50_fpn_backbone, MobileNetV2
from draw_box_utils import draw_box

from xml.dom import minidom


# 将预测值写入xml文件


def ToXml(file,path, photo,dict,predict_classe,category_index):
# 1.创建DOM树对象
    dom = minidom.Document()
# 2.创建根节点。每次都要用DOM对象来创建任何节点。
    root_node = dom.createElement('annotation')
# 3.用DOM对象添加根节点
    dom.appendChild(root_node)

# 用DOM对象创建元素子节点
    folder_node = dom.createElement('folder')
    root_node.appendChild(folder_node)
    folder_text = dom.createTextNode('要标注的')
    folder_node.appendChild(folder_text)

    file_node = dom.createElement('filename')
    root_node.appendChild(file_node)
    file_text = dom.createTextNode(file)
    file_node.appendChild(file_text)

    path_node = dom.createElement('path')
    root_node.appendChild(path_node)
    path_text = dom.createTextNode(path)
    path_node.appendChild(path_text)

    source_node = dom.createElement('source')
    root_node.appendChild(source_node)

    datebase_node = dom.createElement('datebase')
    source_node.appendChild(datebase_node)
    datebase_text = dom.createTextNode('Unknown')
    datebase_node.appendChild(datebase_text)

    size_node = dom.createElement('size')
    root_node.appendChild(size_node)

    width_node = dom.createElement('width')
    size_node.appendChild(width_node)
    width_text = dom.createTextNode(str(photo.size[0]))
    width_node.appendChild(width_text)

    height_node = dom.createElement('height')
    size_node.appendChild(height_node)
    height_text = dom.createTextNode(str(photo.size[1]))
    height_node.appendChild(height_text)

    depth_node = dom.createElement('depth')
    size_node.appendChild(depth_node)
    depth_text = dom.createTextNode('3')
    depth_node.appendChild(depth_text)

    segmented_node = dom.createElement('segmented')
    root_node.appendChild(segmented_node)
    segmented_text = dom.createTextNode('0')
    segmented_node.appendChild(segmented_text)

    for i in range(len(dict)):

        object_node = dom.createElement('object')
        root_node.appendChild(object_node)

        name_node = dom.createElement('name')
        object_node.appendChild(name_node)
        name_text = dom.createTextNode(str(category_index[predict_classe[i]]))
        name_node.appendChild(name_text)

        pose_node = dom.createElement('pose')
        object_node.appendChild(pose_node)
        pose_text = dom.createTextNode('Unspecified')
        pose_node.appendChild(pose_text)

        truncated_node = dom.createElement('truncated')
        object_node.appendChild(truncated_node)
        truncated_text = dom.createTextNode('0')
        truncated_node.appendChild(truncated_text)

        difficult_node = dom.createElement('difficult')
        object_node.appendChild(difficult_node)
        difficult_text = dom.createTextNode('0')
        difficult_node.appendChild(difficult_text)
        #while dict[""]
        bndbox_node = dom.createElement('bndbox')
        object_node.appendChild(bndbox_node)

        xmin_node = dom.createElement('xmin')
        bndbox_node.appendChild(xmin_node)
        xmin_text = dom.createTextNode(str(dict[i][0]))
        xmin_node.appendChild(xmin_text)

        ymin_node = dom.createElement('ymin')
        bndbox_node.appendChild(ymin_node)
        ymin_text = dom.createTextNode(str(dict[i][1]))
        ymin_node.appendChild(ymin_text)

        xmax_node = dom.createElement('xmax')
        bndbox_node.appendChild(xmax_node)
        xmax_text = dom.createTextNode(str(dict[i][2]))
        xmax_node.appendChild(xmax_text)

        ymax_node = dom.createElement('ymax')
        bndbox_node.appendChild(ymax_node)
        ymax_text = dom.createTextNode(str(dict[i][3]))
        ymax_node.appendChild(ymax_text)


# 每一个结点对象（包括dom对象本身）都有输出XML内容的方法，如：toxml()--字符串, toprettyxml()--美化树形格式。

    try:
        with open('./xml/{file}.xml'.format(file=os.path.splitext(file)[0]), 'w', encoding='UTF-8') as fh:
            # 4.writexml()第一个参数是目标文件对象，第二个参数是根节点的缩进格式，第三个参数是其他子节点的缩进格式，
            # 第四个参数制定了换行格式，第五个参数制定了xml内容的编码。
            dom.writexml(fh, indent='', addindent='\t', newl='\n', encoding='UTF-8')
            print('标记成功，已经生成xml文件')
    except Exception as err:
        print('错误：{err}'.format(err=err))

def create_model(num_classes):
    # mobileNetv2+faster_RCNN
    # backbone = MobileNetV2().features
    # backbone.out_channels = 1280
    #
    # anchor_generator = AnchorsGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                     aspect_ratios=((0.5, 1.0, 2.0),))

    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                 output_size=[7, 7],
    #                                                 sampling_ratio=2)
    #
    # model = FasterRCNN(backbone=backbone,
    #                    num_classes=num_classes,
    #                    rpn_anchor_generator=anchor_generator,
    #                    box_roi_pool=roi_pooler)

    # resNet50+fpn+faster_RCNN
    # 注意，这里的norm_layer要和训练脚本中保持一致
    backbone = resnet50_fpn_backbone(norm_layer=torch.nn.BatchNorm2d)
    model = FasterRCNN(backbone=backbone, num_classes=num_classes, rpn_score_thresh=0.5)

    return model


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model 背景加类数
    model = create_model(num_classes=6)

    # load train weights
    train_weights = "./save_weights/resNetFpn-model-4.pth"
    assert os.path.exists(train_weights), "{} file dose not exist.".format(train_weights)
    model.load_state_dict(torch.load(train_weights, map_location=device)["model"])
    model.to(device)

    # read class_indict
    label_json_path = './pascal_voc_classes.json'
    assert os.path.exists(label_json_path), "json file {} dose not exist.".format(label_json_path)
    json_file = open(label_json_path, 'r')
    class_dict = json.load(json_file)
    json_file.close()
    category_index = {v: k for k, v in class_dict.items()}
    path=r'D:\火龙果识别\faster_rcnn\VOCdevkit\VOC2012\test'
    files=os.listdir(path)
    for file in files:
        file_now=os.path.join(path,file)
        original_img = Image.open(file_now)
        # from pil image to tensor, do not normalize image
        data_transform = transforms.Compose([transforms.ToTensor()])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()  # 进入验证模式
        with torch.no_grad():
            # init
            img_height, img_width = img.shape[-2:]
            init_img = torch.zeros((1, 3, img_height, img_width), device=device)
            model(init_img)

            t_start = time_synchronized()
            predictions = model(img.to(device))[0]
            t_end = time_synchronized()
            print("inference+NMS time: {}".format(t_end - t_start))

            predict_boxes = predictions["boxes"].to("cpu").numpy()
            predict_classes = predictions["labels"].to("cpu").numpy()
            predict_scores = predictions["scores"].to("cpu").numpy()

            drop = []
            for i in range(predict_scores.shape[0]):
                if predict_scores[i] < 0.8:
                    drop.append(False)
                else:
                    drop.append(True)
            predict_boxes = predict_boxes[drop]
            predict_classes = predict_classes[drop]
            predict_scores = predict_scores[drop]



            # predict_scores_1=np.transpose(predict_scores)
            # predict = np.concatenate((predict_scores_1,predict_boxes),axis=0)

            # 将预测结果转换成XMl文件
            ToXml(file,file_now,original_img,predict_boxes,predict_classes,category_index)

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(original_img,
                     predict_boxes,
                     predict_classes,
                     predict_scores,
                     category_index,
                     thresh=0.8,
                     line_thickness=3)
            plt.imshow(original_img)
            #plt.show()
            path_out1=os.path.join('./导出/', file)
            # 保存预测的图片结果
            original_img.save(path_out1)
            # return predictions


if __name__ == '__main__':
    main()

