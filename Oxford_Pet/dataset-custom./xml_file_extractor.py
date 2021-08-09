#xml 파일에서 좌표와 파일 이름을 추춣하는 함수
'''
함수 설계
1.xml.etree.ElementTree 임포트 (xml 파일을 읽고 정보를 가져올 수 있는 라이브러리)
2.xml 파일을 인자로 받는 함수 선언
3. tree 변수에 parse 값 입력 ET.parse 사용(xml이 파일로 존재한다면 parse로 받아야 한다.)
,tree로 부터 getroot를 사용해서 root 값을 얻는다.(파일의 최상위 경로)
4. bbox 이름과 bbox 좌표를 보관하는 리스트 객체 선언

5.root.find('object') 루프 실행 -> 이름과 좌표 정보 추출하여 
bbox_names,와 bboxes 정보를 채워 넣는다. -> 두 리스트 객체를 리턴 값으로 반환
'''
import glob
import xml.etree.ElementTree as ET

#annotation xml 파일 파싱해서 bbox정보 추출
def get_bboxes_frm_xml_test(xml_file):
  tree=ET.parse(xml_file)
  root=tree.getroot()
  bbox_names=[]
  bboxes=[]

  for obj in root.finall('object'):

    bbox_name=obj.fine('name').text
    xmlbox=obj.find('bndbox')
    x1=int(xmlbox.find('xmin').text)
    y1=int(xmlbox.find('ymin').text)
    x2=int(xmlbox.find('xmax').text)
    y2=int(xmlbox.find('ymax').text)

    bbox_names.append(bbox_name)
    bboxes.append([x1,y1,x2,y2])

  return bbox_names,bboxes


get_bboxes_from_xml_test('./data/annotations/xmls/Abyssinian_1.xml')
