# Project Motivation
본 프로젝트는 텍스트 데이터를 분석하여 감성을 분류하는 작업에 초점을 맞추었습니다. 특히 영화 리뷰 데이터를 활용하여 텍스트가 긍정적인지 또는 부정적인지를 분류하는 모델을 구축하고자 하였습니다. 이를 통해 자연어 처리(NLP) 기술의 실제 응용 사례를 학습하고, PyTorch와 IMDB 리뷰 데이터셋, Hugging Face Transformers 라이브러리를 활용하여 실질적인 모델 학습 과정을 경험할 수 있었습니다.
# Data Acquisition Method / Model

# Step 1. 디바이스 설정 (Device Configuration)
모델 학습 및 추론 과정에서 GPU를 사용할 수 있도록 디바이스를 설정했습니다. GPU가 없는 경우 CPU를 사용합니다.

# Step 2. 데이터셋 로드 및 샘플링
데이터의 경우 처음에 

dataset = load_dataset("imdb")
train_dataset = dataset['train'].shuffle(seed=42).select(range(100))  # 훈련 데이터 1000개 (추천)
test_dataset = dataset['test'].shuffle(seed=42).select(range(50))   # 테스트 데이터 500개 (추천)

이러한 형식으로 데이터는 IMDB 리뷰 데이터셋으로, Hugging Face Datasets 라이브러리를 통해 가져왔습니다. 이 데이터셋은 영화 리뷰 텍스트와 이에 대한 감성 레이블(positive 또는 negative)을 포함하고 있습니다. 훈련 데이터 1000개, 테스트 데이터 500개를 이용하려 하였지만, 코딩 시간이 늦음을 고려하여 시간상 및 편의성을 위해 정확도를 떨어뜨리지만 시간 효율성을 늘리게끔 훈련 데이터 100개, 테스트 데이터 50개로 줄였다. 훈련 데이터와 테스트 데이터는 shuffle 메서드를 사용해 무작위로 섞은 후 선택했습니다.

# Step 3. 모델 및 토크나이저 초기화 (Model and Tokenizer Initialization)
사용한 모델은 DistilBERT로, 이는 경량화된 BERT 모델입니다. DistilBERT는 빠르고 효율적이면서도 높은 성능을 제공하기 때문에 본 프로젝트에 적합하다 판단하였습니다. Hugging Face Transformers 라이브러리를 통해 모델과 토크나이저를 불러왔습니다. 여기서 함수에 대해서 설명을 추가하면

DistilBertTokenizer: 텍스트를 모델이 이해할 수 있는 토큰 형태로 변환합니다.
DistilBertForSequenceClassification: 입력 텍스트에 대해 감성을 분류합니다. 으로

            tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
            model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2).to(device)
이 형식으로,tokenizer와 model을 지정했습니다.

# Step 4. 데이터 전처리 함수 및 데이터셋 변환 (Data Preprocessing and Transformation)
데이터셋의 텍스트를 토큰화하여 모델 입력으로 변환했습니다. 토큰화 과정에서 패딩, 길이 제한, 잘림(truncation)을 적용했습니다.

                        def preprocess_function(examples):
                                    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)

                                    train_dataset = train_dataset.map(preprocess_function, batched=True)
                                    test_dataset = test_dataset.map(preprocess_function, batched=True)
또한, 데이터셋을 PyTorch 텐서로 변환했습니다

# Step 5. DataLoader 준비 (Preparing DataLoader)
DataLoader를 사용해 배치(batch) 단위로 데이터를 처리할 수 있도록 설정했습니다. 이를 통해 학습 및 평가 단계에서 데이터를 효율적으로 로드했습니다. 이때 batch_size나 num_workers를 조절하여 더욱더 정확도를 높일 수 있습니다.

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=16, collate_fn=collate_fn, num_workers=2)

# Step 6. 옵티마이저와 학습률 스케줄러 설정 (Optimizer and Learning Rate Scheduler)
AdamW 옵티마이저를 사용하여 학습 파라미터를 업데이트했습니다. 학습률은 lr=5e-5로 설정했습니다.
optimizer = AdamW(model.parameters(), lr=5e-5)

# Step 7. 훈련 루프 (Training Loop)
모델 학습을 위한 훈련 루프를 작성했습니다. 각 에폭(epoch)에서 배치 단위로 데이터를 입력받아 손실을 계산하고, 역전파를 통해 모델 파라미터를 업데이트했습니다.
    def train_model(model, train_loader, optimizer, num_epochs):
        model.train()
        for epoch in range(num_epochs):
            total_loss = 0
            for batch in train_loader:
                batch = {key: val.to(device) for key, val in batch.items()}

                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(train_loader):.4f}")
# Step 8. 검증 루프 (Evaluation Loop)
테스트 데이터셋에서 모델의 성능을 평가하기 위해 검증 루프를 작성했습니다. 모델의 예측값과 실제 라벨을 비교하여 정확도(accuracy)를 계산했습니다.
def evaluate_model(model, test_loader):
    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: val.to(device) for key, val in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=-1)

            predictions.extend(preds.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())

    accuracy = accuracy_score(labels, predictions)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Step 9. 훈련 및 평가 (Training and Evaluation)
훈련 루프와 검증 루프를 사용해 모델을 학습시키고 성능을 평가했습니다.

# Step 10. 예측 함수 (Prediction Function)
새로운 텍스트에 대한 감성을 예측할 수 있는 함수를 작성했습니다. 입력 텍스트를 토크나이저를 통해 처리한 뒤, 모델을 사용해 감성 확률과 최종 레이블을 반환합니다.
def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = torch.argmax(probs, dim=-1).item()
        sentiment = "positive" if prediction == 1 else "negative"
        return sentiment, probs.cpu().numpy()
        
# Step 11. 예시 문장 테스트 (Example Sentences Testing)
다양한 문장에 대해 모델이 감성을 정확히 분류하는지 테스트했습니다.
example_text_positive = "The movie was an incredible experience, with a captivating storyline and beautiful performances by the cast."
example_text_negative = "I couldn't stand the movie, it was slow, boring, and lacked any real character development."

for text in [example_text_positive, example_text_negative]:
    sentiment, probs = predict_sentiment(text)
    print(f"Text: {text}\nPredicted Sentiment: {sentiment}, Probabilities: {probs}\n")
    
위와 같은 구조로 진행된 본 프로젝트는 IMDB 리뷰 데이터를 사용하여 텍스트의 긍정 또는 부정 감성을 Positive 혹은 Negative의 표현으로 나타내었습니다.

# Performance

1. 훈련 과정 (Training Process)
훈련 과정에서의 손실(Loss)은 각 에폭(epoch)을 거치며 점진적으로 감소했습니다. 이는 모델이 점점 더 데이터를 잘 학습하고 있음을 나타냅니다.

Epoch 1/3: Loss = 0.6920
Epoch 2/3: Loss = 0.6111
Epoch 3/3: Loss = 0.4422

해석
초기 손실 값(0.6920)은 모델이 랜덤 추측 수준에 머물러 있음을 보여줍니다. 이후 손실 값이 크게 감소했으며, 마지막 에폭에서 0.4422로 도달했습니다. 이는 모델이 텍스트 데이터의 감성 패턴을 효과적으로 학습했음을 의미합니다.

2. 테스트 데이터 평가 (Test Dataset Evaluation)
Test Accuracy: 62.00%

해석
테스트 정확도는 모델이 학습되지 않은 새로운 데이터에서 얼마나 잘 작동하는지를 나타냅니다.
62%라는 정확도는 소규모 데이터(훈련 데이터 100개, 테스트 데이터 50개)를 사용했음을 감안했을 때, 간단한 감성 분석 문제에서는 적절한 성능으로 간주될 수 있습니다.
다만, 실제 사용 사례에 적용하기 위해서는 더 많은 데이터와 추가적인 하이퍼파라미터 튜닝 또는 아키텍처 개선이 필요합니다.

3. 예시 문장 예측 결과 (Example Sentence Predictions)
훈련된 모델을 사용해 두 개의 예시 문장에 대해 감성을 예측했습니다.
각 문장에 대해 감성 레이블과 **확률(probabilities)**을 출력했습니다.

- 문장 1 (Positive)

문장: "The movie was an incredible experience, with a captivating storyline and beautiful performances by the cast."
예측 결과: Positive
확률: [[0.17231593, 0.8276841]]
해석:
긍정 감성(Positive)으로 분류되었으며, 확률 값은 긍정 감성이 **82.76%**로 높은 자신감을 보였습니다.
이는 리뷰에 포함된 긍정적인 단어와 표현("incredible", "captivating", "beautiful")을 모델이 잘 이해했음을 시사합니다.

- 문장 2 (Negative)

문장: "I couldn't stand the movie, it was slow, boring, and lacked any real character development."
예측 결과: Negative
확률: [[0.6052498, 0.39475012]]
해석:
부정 감성(Negative)으로 분류되었으며, 확률 값은 부정 감성이 60.52%, 긍정 감성이 **39.48%**로 나왔습니다.
모델이 "slow", "boring", "lacked"와 같은 부정적인 단어를 인식했으나, 약간의 불확실성도 존재함을 알 수 있습니다.
이는 해당 문장이 복잡한 문맥적 감성을 포함할 경우(예: "부정 표현과 긍정 표현 혼재") 확률이 균등해질 수 있음을 보여줍니다.

4. 모델 성능의 한계와 개선 방향

i) 데이터 크기
사용한 훈련 데이터(100개)와 테스트 데이터(50개)는 매우 제한적입니다.

개선 방안: 더 많은 데이터를 확보하고, 데이터 증강(data augmentation) 기법을 사용해 모델 성능을 향상시킬 수 있습니다.
테스트 정확도

ii) 테스트 정확도
테스트 정확도가 62%로, 단순한 이진 분류 모델의 초기 수준입니다.

개선 방안:
하이퍼파라미터 튜닝(학습률, 배치 크기, 에폭 수 조정).
사전 훈련(pretrained) 모델의 더 큰 버전(DistilBERT 대신 BERT-base 혹은 RoBERTa 등) 사용.
확률 값의 신뢰도

iii) 긍정 문장과 부정 문장의 확률 분포
긍정 문장은 높은 확률로 예측되었으나, 부정 문장은 확률 분포가 상대적으로 덜 명확했습니다.

개선 방안: 더 많은 데이터와 함께 추가적인 훈련 및 정밀한 데이터 전처리를 통해 신뢰도를 높일 수 있습니다.
한국어 모델로의 전환 가능성

결론적으로 본 모델은 제한된 데이터와 간단한 설정으로도 감성 분석 문제를 해결할 수 있음을 보여줬습니다.
특히, 긍정 및 부정 감성에 대해 비교적 정확한 분류 결과를 제공했으며, 확률 값 분석을 통해 문장의 감성을 명확히 표현할 수 있었습니다.
다만, 성능을 더욱 향상시키기 위해 추가적인 데이터, 하이퍼파라미터 튜닝, 또는 모델 아키텍처 개선이 필요합니다.
