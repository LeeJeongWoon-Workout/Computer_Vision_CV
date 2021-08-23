'''
OS와 shutil 두 가지 라이브러리를 통해 파이썬으로 파일과 폴더와 관련된 작업들을 할 수 있다.

OS: 운영체제와 관련된 함수와 클래스를 제공하는 라이브러리로 파일, 폴더와 관련된 함수와 클래스 제공
shutil: 파일, 폴더와 관련된 함수와 클래스를 제공하는 라이브러리
두 라이브러리 모두 파이썬에 기본으로 포함되어 있어 따로 설치할 필요는 없음

'''

import os
import shutil

'''1. 폴더 만들기'''
# 1.1 현재 디렉토리와 파일 리스트 확인

os.getcwd()
!ls

'''2. 파일 쓰기'''
s1 = "data science"
with open("os_dir/test1.txt", "wt") as f:
    f.write(s1)

s2 = "data science2"
with open("os_dir/test2.txt", "wt") as f:
    f.write(s2)

s3 = "data science3"
with open("os_dir/test3.csv", "wt") as f:
    f.write(s3)
    
    
'''3. 파일 리스트 읽기'''
files = os.listdir("os_dir")
files

'''4. 파일 및 폴도 복사하기'''
'''
shutil.copy(원본 경로, 대상 경로)
shutil.copyfile(원본 파일 경로, 대상 파일 경로): 원본이 파일이 아니라 폴더이면 에러 발생
shutil.copytree(원본 폴더 경로, 대상 폴더 경로): 원본이 폴더가 아니라 파일이면 에러 발생
'''

shutil.copy("os_dir/test1.txt", "os_dir/copy1.txt")
shutil.copyfile("os_dir/test1.txt", "os_dir/copy2.txt")
os.listdir("os_dir")
shutil.copytree("os_dir", "os_dir_copy")
!ls




#more information  :  https://hyeshinoh.github.io/2018/10/12/python_09_OS%20&%20shutil/
