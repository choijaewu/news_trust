import torch
from transformers import BertTokenizer, BertForSequenceClassification
import traceback


class BERTClickbaitDetector:
    
    def __init__(self, model_path, model_name='klue/bert-base', max_length=256):
        self.model_path = model_path
        self.model_name = model_name
        self.max_length = max_length
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.tokenizer = None
        
        print(f"Device: {self.device}")
        self._load_model()
    
    def _load_model(self):
        try:
            print(f"Loading model: {self.model_path}")
            
            # 체크포인트 로드
            try:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device,
                    weights_only=False
                )
            except TypeError:
                checkpoint = torch.load(
                    self.model_path,
                    map_location=self.device
                )
            
            # 체크포인트에서 설정 추출
            self.model_name = checkpoint.get('model_name', self.model_name)
            self.max_length = checkpoint.get('max_length', self.max_length)
            model_state = checkpoint.get('model_state_dict', checkpoint)
            
            print(f"Model: {self.model_name}, Max length: {self.max_length}")
            
            # 토크나이저 초기화
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            
            # 모델 초기화
            self.model = BertForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=2
            )
            
            # 학습된 가중치 로드
            try:
                self.model.load_state_dict(model_state, strict=False)
                print("Model weights loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load weights, using pretrained only: {e}")
            
            self.model.to(self.device)
            self.model.eval()
            
            print("Model ready")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise
    
    def predict(self, title, content=""):
        try:
            # 제목과 본문 결합
            text = f"{title} [SEP] {content}"
            
            # 토크나이징
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            # 예측
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits  # 원시점수(+,-) -> 확률로 변환 필요
                probs = torch.softmax(logits, dim=1)
                pred = torch.argmax(probs, dim=1) 
            
            pred_label = pred.item()  # 정수로 변환
            probs_cpu = probs[0].cpu().numpy()  #gpu 텐서 -> cpu 텐서 -> numpy 배열
            
            return {
                'is_clickbait': pred_label == 1,
                'clickbait_probability': float(probs_cpu[1]),
                'confidence': float(probs_cpu[pred_label]),
                'method': 'BERT'
            }
            
        except Exception as e:
            print(f"Prediction error: {e}")
            traceback.print_exc()
            # 오류 시 기본값 반환
            return {
                'is_clickbait': False,
                'clickbait_probability': 0.5,
                'confidence': 0.5,
                'method': 'error'
            }