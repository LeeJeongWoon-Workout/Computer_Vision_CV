#xml 파일에서 좌표와 파일 이름을 추춣하는 함수
'''
설계
1.get_bboxes_from_xml(anno_dir,xml_file)
osp.join 으로 anno_dir,xml_file을 합쳐 절대 경로를 만든다.
2. 절대 경로를 기준으로 tree,root 객체를 생성한다.
3. bbox_names,bboxes 리스트 객체 생성
4. loop (root.findall('object))
5. 클레스 명은 파일명에 있으므로 xml_file[:xml_file.rfind('_')] 문자열 함수 이용
6. 나머지 'bndbox'에서 좌표 추출
7. bbox_names,bboxes 반환
'''

import xml.etree.ElementTree as ET
import os.path as osp

def get_bboxes_from_xml(anno_dir,xml_file):
  anno_xml_file=osp.join(anno_dir,xml_file) #절대 경로
  tree=ET.parse(anno_xml_file)
  root=tree.getroot()
  bbox_names=[]
  bboxes=[]

  for obj in root.findall('object'):
    #obj.find('name).text는 cat이나 dog를 반환
    #object 클래스명은 파일명에서 추출
    bbox_name=xml_file[:xml_file.rfind('_')]

    xmlbox=obj.find('bndbox')

    x1=int(xmlbox.find('xmin').text)
    y1=int(xmlbox.find('ymin').text)
    x2=int(xmlbox.find('xmax').text)
    y2=int(xmlbox.find('ymax').text)

    bbox_names.append(bbox_name)
    bboxes.append([x1,y1,x2,y2])
  
  return bbox_names,bboxes
