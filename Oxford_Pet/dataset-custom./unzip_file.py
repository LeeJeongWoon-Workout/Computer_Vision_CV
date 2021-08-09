!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
!wget https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz

# /content/data 디렉토리를 만들고 해당 디렉토리에 다운로드 받은 압축 파일 풀기.
!mkdir /content/data  #pet 데이터를 보관한 directory 생성
!tar -xvf images.tar.gz -C /content/data  #tar -xvf(압축을 푼다) ~(~파일을) -C/content/data (-C ~ 위치에)
!tar -xvf annotations.tar.gz -C /content/data
