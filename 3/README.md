

##심화
## Q1) 어떤 task를 선택하셨나요?
> NER, MNLI, 기계 번역 셋 중 하나를 선택
MNLI 

## Q2) 모델은 어떻게 설계하셨나요? 설계한 모델의 입력과 출력 형태가 어떻게 되나요?
> 모델의 입력과 출력 형태 또는 shape을 정확하게 기술
3가지 분류형태로 2진분류가 아닌 그냥 분류로 했습니다.

## Q3) 실제로 pre-trained 모델을 fine-tuning했을 때 loss curve은 어떻게 그려지나요? 그리고 pre-train 하지 않은 Transformer를 학습했을 때와 어떤 차이가 있나요? 
> 비교 metric은 loss curve, accuracy, 또는 test data에 대한 generalization 성능 등을 활용.
> +)이외에도 기계 번역 같은 문제에서 활용하는 BLEU 등의 metric을 마음껏 활용 가능
- pre-trained모델을 fine-tuning했을때랑 그냥 했을때랑 정확도에 차이가 있었습니다.
-  
-  
- 이미지 첨부시 : ![이미지 설명](경로) / 예시: ![poster](./image.png)

### 위의 사항들을 구현하고 나온 결과들을 정리한 보고서를 README.md 형태로 업로드
### 코드 및 실행 결과는 jupyter notebook 형태로 같이 public github repository에 업로드하여 공유해주시면 됩니다. 반드시 출력 결과가 남아있어야 합니다.
