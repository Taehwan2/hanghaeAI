기본과제 wandb 
https://wandb.ai/taehwanjj-kfr/Hanghae99/reports/train-loss-25-05-14-20-46-55---VmlldzoxMjc3MDk1Mg




심화과제 wandb
https://wandb.ai/taehwanjj-kfr/Hanghae99/reports/train-loss-25-05-14-23-37-28---VmlldzoxMjc3MzgyMg

lean.py => 모델 학습하는 코드 basket.json을 불러와서 학습
start.py => instruction tuning만 사용하여 결과도출
확실하게 instruction 튜닝한 질문 그대로하면 가끔은 정확하게 맞추지만, 가끔 틀릴 때가 있음 => 모델의 크기와 학습 에포크가 적어서 그런거같음
hard.py => lag와 instruct tuning을 같이 사용하여 결과 도출 => 확실히 instruction tuning과 rag를 사용해서 하니깐 답변다운 답변이 나옴 GPT에게 점수를 먹여달라니깐 정확도는 높지만, 요약이 부족하다고 나옴
