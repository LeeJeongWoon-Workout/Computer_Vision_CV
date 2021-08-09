'''현재 trainval 에 다 모여 있기 때문에 pandas를 이용해 
csv 파일로 만든후 sklear.model_selection으로 나눈 다음 to_csv를 통해
다시 train 용 val 용 데이터를 담는 txt 메타 파일을 만들 것이다.'''
'''
과정
1.pd.read_csv(파일 경로,무슨 기준으로 정보들을 나눌 것인가,헤더,열 이름들 리스트로 입력)
2.apply(lambda로) class_name을 담는 새로운 column 생성
3. sklearn.model_selection, train_test_split로 train_df와 val_df 를 만든다.
4. sort_values(by='img_name')
5. ~['img_name'].to_csv('./data/train.txt',~) 로 훈련용,val용 txt 파일을 만든다.
'''

import pandas as pd

pet_df=pd.read_csv('./data/annotations/trainval.txt',sep=' ',header=None,names=['img_name','class_id','etc1','ect2'])
pet_df['class_name']=pet_df['img_name'].apply(lambda x:x[:x.rfind('_')])
pet_df.head()
pet_df=pet_df.sort_values(by='img_name')

from sklearn.model_selection import train_test_split

train_df,val_df=train_test_split(pet_df,test_size=0.1)
#train옹 valid용 메타파일을 따로 만드는 과정
train_df['img_name'].to_csv('./data/train.txt',sep=' ',header=False,index=False)
val_df['img_name'].to_csv('./data/val.txt',sep=' ',header=False,index=False)

!echo 'train list #####';cat ./data/train.txt
!echo 'val list #####'; cat ./data/val.txt
