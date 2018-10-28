# Windows에서 Python running 환경 설정

## conda version
```
(base) C:\Users\User> conda --version
```

## conda 설치 위치 확인
```
(base) C:\Users\User> where conda
```

## 패키지 설치 목록
```
(base) C:\Users\User> conda list
```

## Upgrade Conda
```
(base) C:\Users\User> conda upgrade conda
(base) C:\Users\User> conda upgrade --all
```
## Tensorflow 설치
```
(base) C:\Users\User> pip install --ignore-installed--upgrade tensorflow
```

## Tree Map을 그리기 위한 graphviz 설치(Windows환경)
```
(base) C:\Users\User> pip install graphviz
```

## graphviz.msi 다운로드 + 설치
-  link : https://graphviz.gitlab.io/_pages/Download/windows/graphviz-2.38.msi

- graphviz-2.38의 설치 딜렉토리와 환경설정
  - 정확하게 설치가 되지 않으면 Decision Tree Graph를 조회 오류가 발생합니다.
  - 설치 Directory 추천 : C:\Apps\graphviz-2.38
  - windows path : C:\Apps\graphviz-2.38\bin

