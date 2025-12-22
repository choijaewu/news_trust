import nltk
from newspaper import Article
import streamlit as st
from emotion import emotion_analyze
from clickbait_detector import BERTClickbaitDetector
import torch
import torch.nn as nn
import numpy as np
import re

@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')

# 앱 시작 시 다운로드
download_nltk_data()

# 페이지 설정
st.set_page_config(
    page_title="뉴스 품질 평가",
    page_icon="📰",
    layout="wide"
)

# 토크나이저
class SimpleTokenizer:
    def morphs(self, text):
        return re.findall(r'\w+', text)

tokenizer = SimpleTokenizer()

# 요약 모델 정의
class FastSummarizer(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim//2, bidirectional=True, 
                           batch_first=True, num_layers=1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, n, s = x.size()
        x = x.view(b * n, s)
        emb = self.embedding(x)
        mask = (x != 0).unsqueeze(-1).float()
        sent_repr = (emb * mask).sum(1) / (mask.sum(1) + 1e-8)
        sent_repr = sent_repr.view(b, n, -1)
        doc_repr, _ = self.lstm(sent_repr)
        scores = self.fc(doc_repr).squeeze(-1)
        return scores

# 캐시된 요약 모델 로드
@st.cache_resource
def load_summarizer_model():
    try:
        vocab = torch.load('vocab.pt')
        model = FastSummarizer(len(vocab), embed_dim=100, hidden_dim=128)
        model.load_state_dict(torch.load('fast_model.pt', map_location='cpu'))
        model.eval()
        return model, vocab
    except:
        return None, None

# 중요 문장 예측
def predict_key_sentences(model, vocab, text, top_k=3, max_sent_len=30, max_doc_len=20):
    if model is None or vocab is None:
        return []
    
    try:
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        
        if len(sentences) < 3:
            return []
        
        content_sentences = sentences[:max_doc_len]
        
        # 인코딩
        encoded = []
        for sent in content_sentences:
            tokens = tokenizer.morphs(str(sent))[:max_sent_len]
            indices = [vocab.get(t, 1) for t in tokens]
            indices += [0] * (max_sent_len - len(indices))
            encoded.append(indices)
        
        num_sents = len(encoded)
        while len(encoded) < max_doc_len:
            encoded.append([0] * max_sent_len)
        
        with torch.no_grad():
            x = torch.LongTensor(encoded).unsqueeze(0)
            scores = model(x).squeeze().numpy()
        
        # 상위 k개 문장 선택
        valid_scores = scores[:num_sents]
        top_idx = np.argsort(valid_scores)[-top_k:][::-1]
        
        return [content_sentences[i] for i in top_idx]
    except:
        return []

# 기사에서 텍스트 추출
def get_article_text(url):
    try:
        article = Article(url, language='ko')
        article.download()
        article.parse()
        return article.title, article.text
    except Exception as e:
        st.error(f"기사 추출 실패: {e}")
        return None, None

# 점수 계산
def calculate_positive_scores(clickbait_prob, emotion_score):
    reliability_score = max(0, 100 - clickbait_prob)
    objectivity_score = max(0, 100 - emotion_score)
    quality_score = reliability_score * 0.7 + objectivity_score * 0.3
    
    return reliability_score, objectivity_score, quality_score

# 뉴스 신뢰등급 평가
def get_quality_grade(score):
    if score >= 85:
        return "🏆 최우수", "success"
    elif score >= 70:
        return "🥇 우수", "success"
    elif score >= 55:
        return "🥈 양호", "warning"
    elif score >= 40:
        return "🥉 보통", "warning"
    else:
        return "⚠️ 주의", "error"


# Streamlit UI
st.title("📰 뉴스 품질 평가 시스템")
st.caption("BERT 기반 AI가 뉴스의 신뢰도와 객관성을 종합적으로 평가합니다")

# 사이드바
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.write("""
    1. 뉴스 기사 URL을 입력하세요
    2. '품질 분석하기' 버튼을 클릭하세요
    3. 종합 점수와 세부 분석을 확인하세요
    
    **✨ 평가 기준:**
    - **신뢰도 (70%)**: 낚시성이 낮을수록 높은 점수
    - **객관성 (30%)**: 감정적 표현이 적을수록 높은 점수
    """)
    
    st.header("🏆 등급 기준")
    st.write("""
    - **최우수 (85-100점)**: 🏆
    - **우수 (70-84점)**: 🥇
    - **양호 (55-69점)**: 🥈
    - **보통 (40-54점)**: 🥉
    - **주의 (0-39점)**: ⚠️
    """)

# 요약 모델 로드
summary_model, vocab = load_summarizer_model()

# 세션 상태 초기화
if 'analysis_done' not in st.session_state:
    st.session_state.analysis_done = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# 메인 입력 영역
url = st.text_input(
    "뉴스 기사 URL 입력",
    placeholder="https://news.example.com/article/12345",
    help="네이버, 다음 등의 뉴스 기사 URL을 입력하세요"
)

if st.button("🔍 품질 분석하기", type="primary"):
    if url:
        with st.spinner('기사를 분석하는 중입니다...'):
            try:
                # 기사 추출
                st.info("📰 기사 텍스트 추출 중...")
                title, text = get_article_text(url)
                
                if not title or not text:
                    st.error("기사를 추출할 수 없습니다. URL을 확인해주세요.")
                    st.stop()
                else:
                    st.success("✅ 텍스트 추출 완료!")
                
                # 낚시성 분석
                st.info("🎣 신뢰도 분석 중...")
                detector = BERTClickbaitDetector(model_path="clickbait_detector_bert1.pt", model_name='klue/bert-base', max_length=256)
                result = detector.predict(title, text)
                clickbait_prob = result['clickbait_probability'] * 100
                st.success("✅  신뢰도 분석 완료!")
                
                # 감정 분석
                st.info("😊 객관성 분석 중...")
                emotion, e_score = emotion_analyze(text)
                st.success("✅  객관성 분석 완료!")
                
                # 중요 문장 추출
                key_sentences = []
                if summary_model is not None:
                    st.info("✨ 중요 문장 추출 중...")
                    key_sentences = predict_key_sentences(summary_model, vocab, text)
                    st.success("✅ 중요 문장 추출 완료!")
                
                # 점수 계산
                reliability_score, objectivity_score, quality_score = calculate_positive_scores(
                    clickbait_prob, e_score
                )
                
                # 결과 저장 
                st.session_state.results = {
                    'title': title,
                    'text': text,
                    'clickbait_prob': clickbait_prob,
                    'emotion': emotion,
                    'e_score': e_score,
                    'reliability_score': reliability_score,
                    'objectivity_score': objectivity_score,
                    'quality_score': quality_score,
                    'url': url,
                    'method': result['method'],
                    'key_sentences': key_sentences
                }
                st.session_state.analysis_done = True
                st.success("✅ 분석 완료!")
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ 오류 발생: {e}")
                st.session_state.analysis_done = False
    else:
        st.warning("URL을 입력해주세요.")


if st.session_state.analysis_done and st.session_state.results:
    results = st.session_state.results
    
    st.divider()
    st.header("📊 품질 분석 결과")
    
    # 메인 품질 점수
    quality_grade, grade_type = get_quality_grade(results['quality_score'])
    
    col_main = st.columns(1)[0]
    with col_main:
        if grade_type == "success":
            st.success(f"**종합 품질 점수: {results['quality_score']:.0f}점** {quality_grade}")
        elif grade_type == "warning":
            st.warning(f"**종합 품질 점수: {results['quality_score']:.0f}점** {quality_grade}")
        else:
            st.error(f"**종합 품질 점수: {results['quality_score']:.0f}점** {quality_grade}")
    
    # 세부 점수
    st.subheader("🏅 세부 평가 점수")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        reliability_delta = "높음" if results['reliability_score'] > 70 else "보통" if results['reliability_score'] > 50 else "낮음"
        st.metric(
            "🛡️ 신뢰도 점수",
            f"{results['reliability_score']:.0f}점",
            delta=reliability_delta,
            delta_color="normal"
        )
    
    with col2:
        objectivity_delta = "높음" if results['objectivity_score'] > 70 else "보통" if results['objectivity_score'] > 50 else "낮음"
        st.metric(
            "⚖️ 객관성 점수",
            f"{results['objectivity_score']:.0f}점",
            delta=objectivity_delta,
            delta_color="normal"
        )
    
    with col3:
        st.metric(
            "📈 종합 품질",
            f"{results['quality_score']:.0f}점",
            delta=quality_grade.split()[1] if len(quality_grade.split()) > 1 else "",
            delta_color="normal"
        )
    
    # 상세 분석
    st.subheader("📋 상세 분석")
    
    col_detail1, col_detail2 = st.columns(2)
    
    with col_detail1:
        st.write("**🎣 낚시성 분석**")
        if results['clickbait_prob'] < 30:
            st.success(f"낚시성 확률: {results['clickbait_prob']:.1f}% (신뢰할 만함)")
        elif results['clickbait_prob'] < 70:
            st.warning(f"낚시성 확률: {results['clickbait_prob']:.1f}% (주의 필요)")
        else:
            st.error(f"낚시성 확률: {results['clickbait_prob']:.1f}% (높은 주의)")
    
    with col_detail2:
        st.write("**😊 감정 분석**")
        st.write(f"감정 유형: **{results['emotion']}**")
        if results['e_score'] < 30:
            st.success(f"감정 강도: {results['e_score']:.1f}점 (매우 객관적)")
        elif results['e_score'] < 70:
            st.warning(f"감정 강도: {results['e_score']:.1f}점 (적당히 감정적)")
        else:
            st.error(f"감정 강도: {results['e_score']:.1f}점 (매우 감정적)")
    
    # 기사 내용
    st.subheader("📰 기사 내용")
    st.write(f"**제목**: {results['title']}")
    st.write(f"**출처**: {results['url']}")
    
    with st.expander("📖 기사 본문 보기 (중요 문장 하이라이트)"):
        # 문장 분리
        sentences = re.split(r'(?<=[.!?])\s+', results['text'])
        key_sentences = results.get('key_sentences', [])
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 5:
                # 중요 문장인지 확인
                is_key = any(sent in key_sent for key_sent in key_sentences)
                
                if is_key:
                    st.markdown(f"""
                    <div style="background-color: #fef3c7; padding: 10px; 
                                border-radius: 5px; border-left: 4px solid #f59e0b;
                                margin-bottom: 8px;">
                        <strong style="color: #92400e;">⭐ {sent}</strong>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #f9fafb; padding: 10px; 
                                border-radius: 5px; margin-bottom: 8px; color: #374151;">
                        {sent}
                    </div>
                    """, unsafe_allow_html=True)
    

# 푸터
st.divider()
st.caption("🤖 BERT 기반 AI 모델 | 🔬 딥러닝 기술 사용 | ⚡ Powered by PyTorch & Transformers")