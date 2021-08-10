import xml.etree.ElementTree as ET
import os.path as osp

def get_bboxes_from_xml(anno_dir,xml_file):
  anno_xml_file=osp.join(anno_dir,xml_file) #절대 경로
  tree=ET.parse(anno_xml_file)
  root=tree.getroot()
  bbox_names=[]
  bboxes=[]

  for obj in root.findall('object'):
    #xml 파일 안에 클래스 명이 들어가 있음.
    bbox_name=obj.find('name').text

    xmlbox=obj.find('bndbox')

    x1=int(xmlbox.find('xmin').text)
    y1=int(xmlbox.find('ymin').text)
    x2=int(xmlbox.find('xmax').text)
    y2=int(xmlbox.find('ymax').text)

    bbox_names.append(bbox_name)
    bboxes.append([x1,y1,x2,y2])
  
  return bbox_names,bboxes
