import streamlit as st
from emotion import emotion_analyze
from clickbait_detector import BERTClickbaitDetector
import torch
import torch.nn as nn
import numpy as np
import re

# 페이지 설정
st.set_page_config(
    page_title="AI 뉴스 품질 플랫폼",
    page_icon="📰",
    layout="wide"
)

# ------------------------------------------------------------------------------
# 사용자 정의 CSS (뉴스 사이트 스타일)
# ------------------------------------------------------------------------------
st.markdown("""
<style>
/* Streamlit 기본 스타일 조정 */
.stApp { padding-top: 0 !important; }
.css-h5fmrh, .css-1dp17k9 { padding-top: 0rem; padding-bottom: 0rem; }
h1, h2, h3, h4, h5, h6 { color: #1a1a1a; }

/* 메인 헤더 영역 스타일 */
.main-header {
    background-color: #f0f2f6;
    padding: 20px 0;
    margin-bottom: 20px;
    border-bottom: 3px solid #1f77b4;
}
.main-header h1 {
    color: #1f77b4;
    font-weight: 700;
    margin: 0;
    display: inline-block;
}

/* 기사 카드 스타일 */
.article-card {
    background-color: white;
    border: 1px solid #eee;
    border-radius: 8px;
    padding: 20px;
    margin-bottom: 15px;
    cursor: pointer;
    transition: all 0.2s ease;
}

.article-card:hover {
    background-color: #f7f9fb;
    border-color: #1f77b4;
    box-shadow: 0 2px 8px rgba(31, 119, 180, 0.1);
}

.article-card.selected {
    border-left: 5px solid #1f77b4;
    background-color: #f0f7ff;
}

.article-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 12px;
}

.article-title {
    font-size: 20px;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
    flex: 1;
    text-align: left;
    line-height: 1.4;
}

.expand-icon {
    font-size: 24px;
    color: #1f77b4;
    margin-left: 15px;
    font-weight: bold;
    cursor: pointer;
    user-select: none;
}

.article-scores {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}

/* 점수 태그 공통 스타일 */
.score-tag {
    color: white;
    padding: 6px 12px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
    display: inline-block;
}

/* 본문 영역 스타일 */
.article-body-container {
    padding: 20px;
    background-color: #f9fafb;
    border: 1px solid #ddd;
    border-radius: 8px;
    margin-top: 15px;
}
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# 모델 및 함수 정의
# ------------------------------------------------------------------------------

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
    except Exception as e:
        return None, None

# 중요 문장 예측
def predict_key_sentences(model, vocab, text, top_k=3, max_sent_len=30, max_doc_len=20):
    if model is None or vocab is None: return []
    try:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
        if len(sentences) < 3: return []
        content_sentences = sentences[:max_doc_len]
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
        valid_scores = scores[:num_sents]
        top_idx = np.argsort(valid_scores)[-top_k:][::-1]
        return [content_sentences[i] for i in top_idx]
    except Exception as e:
        return []

# 점수 계산
def calculate_positive_scores(clickbait_prob, emotion_score):
    reliability_score = max(0, 100 - clickbait_prob) 
    objectivity_score = max(0, 100 - emotion_score) 
    quality_score = reliability_score * 0.7 + objectivity_score * 0.3
    return reliability_score, objectivity_score, quality_score

# 신뢰 등급 평가
def get_quality_grade(score):
    if score >= 85: return "🏆 최우수", "success"
    elif score >= 70: return "🥇 우수", "success"
    elif score >= 55: return "🥈 양호", "warning"
    elif score >= 40: return "🥉 보통", "warning"
    else: return "⚠️ 주의", "error"

# 점수 색깔을 다르게 표시
def get_score_color(score, type='quality'):
    score = int(score)
    if type == 'quality':
        if score >= 85: return "#10B981"
        if score >= 70: return "#34D399"
        if score >= 55: return "#FCD34D"
        if score >= 40: return "#FB923C"
        return "#EF4444"
    elif type == 'clickbait_prob' or type == 'emotion_score':
        if score <= 30: return "#10B981"
        if score <= 50: return "#34D399"
        if score <= 70: return "#FCD34D"
        if score <= 90: return "#FB923C"
        return "#EF4444"
    return "black"

# ------------------------------------------------------------------------------
# 세션 상태 초기화
# ------------------------------------------------------------------------------
if 'news_articles' not in st.session_state:
    st.session_state.news_articles = []
if 'selected_article_idx' not in st.session_state:
    st.session_state.selected_article_idx = -1
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'show_editor' not in st.session_state:
    st.session_state.show_editor = False
if 'analysis_done_temp' not in st.session_state:
    st.session_state.analysis_done_temp = False

# ------------------------------------------------------------------------------
# 메인 UI 구성
# ------------------------------------------------------------------------------

# 헤더 영역
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.markdown("<h1>📰 AI 뉴스 품질 플랫폼</h1>", unsafe_allow_html=True)
st.caption("작성된 뉴스 기사의 신뢰도, 낚시성, 감정적 점수를 분석하고 게시합니다.")
st.markdown('</div>', unsafe_allow_html=True)

# 모델 로드
summary_model, vocab = load_summarizer_model()
if summary_model is None and not st.session_state.get('model_warning_shown', False):
    st.warning("경고: 요약 모델(fast_model.pt, vocab.pt) 로드에 실패했습니다. 중요 문장 추출 기능은 작동하지 않습니다.")
    st.session_state.model_warning_shown = True

# ==============================================================================
# 뉴스 작성 버튼 및 모달 창 표시 제어
# ==============================================================================

main_content = st.container()

with main_content:
    
    col_top_bar = st.columns([1, 4])
    
    if col_top_bar[0].button("📝 뉴스 기사 작성하기", type="primary"):
        st.session_state.show_editor = True
        st.session_state.analysis_done_temp = False
        st.session_state.analysis_results = None
        st.session_state.selected_article_idx = -1
        st.rerun()
    
    col_top_bar[1].markdown(f"**총 {len(st.session_state.news_articles)}건**의 기사가 등록되어 있습니다.", unsafe_allow_html=True)
    
    st.markdown("---")


# ==============================================================================
# 뉴스 작성 및 분석 (Editor)
# ==============================================================================
if st.session_state.show_editor:
    
    st.header("새 기사 등록 및 품질 분석")
    
    with st.form(key='article_form'):
        title = st.text_input("뉴스 기사 제목 입력", key="new_article_title")
        text = st.text_area("뉴스 기사 본문 입력", height=300, key="new_article_text")
        
        col_buttons = st.columns([1, 1, 4])
        analyze_button = col_buttons[0].form_submit_button("🔍 품질 분석하기", type="primary")
        cancel_button = col_buttons[1].form_submit_button("취소")

    if cancel_button:
        st.session_state.show_editor = False
        st.session_state.analysis_done_temp = False
        st.session_state.analysis_results = None
        st.rerun()

    if analyze_button:
        if not title: st.warning("기사 제목을 입력해주세요.")
        elif len(text) < 100: st.warning("기사 본문이 너무 짧습니다. 100자 이상 입력해주세요.")
        else:
            with st.spinner('기사를 분석하는 중입니다...'):
                try:
                    # 분석 로직 (외부 모듈을 사용한다고 가정)
                    detector = BERTClickbaitDetector(model_path="clickbait_detector_bert1.pt", model_name='klue/bert-base', max_length=256)
                    result = detector.predict(title, text)
                    clickbait_prob = result['clickbait_probability'] * 100
                    emotion, e_score = emotion_analyze(text)
                    key_sentences = predict_key_sentences(summary_model, vocab, text)
                    reliability_score, objectivity_score, quality_score = calculate_positive_scores(clickbait_prob, e_score)
                    
                    st.session_state.analysis_results = {
                        'title': title, 'text': text, 'clickbait_prob': clickbait_prob, 
                        'emotion': emotion, 'e_score': e_score, 'reliability_score': reliability_score, 
                        'objectivity_score': objectivity_score, 'quality_score': quality_score, 
                        'key_sentences': key_sentences
                    }
                    st.session_state.analysis_done_temp = True
                    st.success(f"✅ 분석 완료! 종합 점수: {quality_score:.0f}점")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 분석 중 오류 발생. 외부 모델 파일(BERTClickbaitDetector, emotion_analyze)을 확인해주세요: {e}")
                    st.session_state.analysis_results = None
                    st.session_state.analysis_done_temp = False

    # 2단계: 분석 결과 표시 및 최종 등록 확인
    if st.session_state.analysis_done_temp and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        st.divider()
        st.subheader("💡 분석된 기사 등록 준비 (최종 확인)")
        
        quality_grade, _ = get_quality_grade(results['quality_score'])
        
        st.markdown(f"#### **{results['title']}**")
        st.markdown(f"**종합 신뢰도**: <span style='color:{get_score_color(results['quality_score'], 'quality')}; font-size:18px;'>**{results['quality_score']:.0f}점 ({quality_grade.split()[0]})**</span>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("낚시성 점수 (낮을수록 좋음)", f"{results['clickbait_prob']:.0f}점")
        with col2:
            st.metric("감정 강도 점수 (낮을수록 좋음)", f"{results['e_score']:.0f}점")
        
        if st.button("✅ 최종 확인 후 등록하기", key="final_register_button", type="primary"):
            st.session_state.news_articles.append(results)
            st.session_state.analysis_results = None
            st.session_state.analysis_done_temp = False
            st.session_state.show_editor = False
            st.success(f"'{results['title']}' 기사가 성공적으로 등록되었습니다!")
            st.rerun()
            
# ==============================================================================
# 등록된 기사 목록 (List View) 및 클릭 로직 - 컬럼 방식
# ==============================================================================

if not st.session_state.show_editor:
    
    st.subheader("최신 기사 목록")

    if not st.session_state.news_articles:
        st.info("등록된 기사가 없습니다. '뉴스 기사 작성하기' 버튼을 눌러 새로운 기사를 등록해주세요.")
    else:
        
        displayed_articles = st.session_state.news_articles[::-1] 
        
        for i, article in enumerate(displayed_articles):
            original_idx = len(st.session_state.news_articles) - 1 - i # 실제 인덱스
            
            quality_score = article['quality_score']
            fishing_prob = article['clickbait_prob'] 
            emotion_score = article['e_score'] 
            
            # 점수별 색상 계산
            q_color = get_score_color(quality_score, 'quality')
            f_color = get_score_color(fishing_prob, 'clickbait_prob')
            e_color = get_score_color(emotion_score, 'emotion_score')
            
            is_selected = st.session_state.selected_article_idx == original_idx
            
            # 토글 아이콘
            indicator = '−' if is_selected else '∔'
            
            # 클릭 이벤트 정의
            def set_selected_article(idx):
                if st.session_state.selected_article_idx == idx:
                    st.session_state.selected_article_idx = -1
                else:
                    st.session_state.selected_article_idx = idx

            # 기사 카드 HTML
            card_class = "article-card selected" if is_selected else "article-card"
            article_html = f"""
            <div class="{card_class}">
                <div class="article-header">
                    <div class="article-title">{article['title']}</div>
                </div>
                <div class="article-scores">
                    <span class="score-tag" style="background-color: {q_color};">
                        신뢰도 {quality_score:.0f}점
                    </span>
                    <span class="score-tag" style="background-color: {f_color};">
                        낚시성 {fishing_prob:.0f}점
                    </span>
                    <span class="score-tag" style="background-color: {e_color};">
                        감정적 {emotion_score:.0f}점
                    </span>
                </div>
            </div>
            """
            
            # 컬럼으로 배치: 카드와 버튼을 나란히
            col1, col2 = st.columns([20, 1])
            
            with col1:
                st.markdown(article_html, unsafe_allow_html=True)
            
            with col2:
                if st.button(indicator, key=f"toggle_{original_idx}", help="기사 전문 보기"):
                    set_selected_article(original_idx)
                    st.rerun()

            # 선택된 기사 본문 표시
            if is_selected:                
                sentences = re.split(r'(?<=[.!?])\s+', article['text'])
                key_sentences = article.get('key_sentences', [])
                
                # 본문 내용 하이라이트 로직
                for sent in sentences:
                    sent = sent.strip()
                    is_key = any(sent in key_sent or key_sent in sent for key_sent in key_sentences)
                    
                    if len(sent) > 5:
                        if is_key:
                            st.markdown(f"""
                            <div style="background-color: #fef3c7; padding: 12px; 
                                        border-radius: 6px; border-left: 4px solid #f59e0b;
                                        margin-bottom: 10px;">
                                <strong style="color: #92400e;">⭐ {sent}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div style="background-color: white; padding: 12px; 
                                        border-radius: 6px; margin-bottom: 10px; color: #374151;
                                        line-height: 1.6;">
                                {sent}
                            </div>
                            """, unsafe_allow_html=True)


# 푸터
st.divider()
st.caption("🤖 BERT 기반 AI 모델 | 🔬 딥러닝 기술 사용 | ⚡ Powered by Streamlit")