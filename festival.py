import streamlit as st
from clickbait_detector import BERTClickbaitDetector
import random

# 페이지 설정
st.set_page_config(
    page_title="나도, 기자?",
    page_icon="📝",
    layout="wide"
)

# 사용자 정의 CSS
st.markdown("""
<style>
.stApp { padding-top: 0 !important; }
h1, h2, h3 { color: #1a1a1a; }

.game-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    margin-bottom: 30px;
    text-align: center;
    color: white;
}

.game-header h1 {
    color: white;
    margin: 0;
    font-size: 2.5em;
}

.title-card {
    background-color: #f8f9fa;
    border-left: 5px solid #667eea;
    padding: 25px;
    border-radius: 10px;
    margin: 20px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
}

.title-text {
    font-size: 24px;
    font-weight: 600;
    color: #1a1a1a;
    margin: 0;
}

.score-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 30px;
    border-radius: 15px;
    text-align: center;
    color: white;
    margin: 20px 0;
}

.score-value {
    font-size: 90px;
    font-weight: bold;
    margin: 10px 0;
}

.score-label {
    font-size: 24px;
    opacity: 0.9;
}

.result-message {
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    text-align: center;
    font-size: 18px;
    font-weight: 600;
}

.excellent { background-color: #d4edda; color: #155724; border: 2px solid #c3e6cb; }
.good { background-color: #d1ecf1; color: #0c5460; border: 2px solid #bee5eb; }
.average { background-color: #fff3cd; color: #856404; border: 2px solid #ffeaa7; }
.poor { background-color: #f8d7da; color: #721c24; border: 2px solid #f5c6cb; }

.info-box {
    background-color: #e7f3ff;
    border-left: 4px solid #2196F3;
    padding: 15px;
    border-radius: 5px;
    margin: 15px 0;
}

.article-display {
    background-color: #f8f9fa;
    padding: 20px;
    border-radius: 10px;
    margin: 20px 0;
    border: 1px solid #dee2e6;
}

.article-display h4 {
    color: #667eea;
    margin-bottom: 10px;
}

.article-content {
    background-color: white;
    padding: 15px;
    border-radius: 5px;
    line-height: 1.6;
    color: #333;
    white-space: pre-wrap;
    word-wrap: break-word;
}
</style>
""", unsafe_allow_html=True)

# 뉴스 제목 리스트
NEWS_TITLES = [
    "대통령, 긴급 기자회견 통해 경제정책 발표",
    "서울 집값 급등, 정부 대책 마련 나서",
    "K-POP 스타, 빌보드 차트 1위 달성",
    "인공지능 기술, 의료 현장에 혁신 가져와",
    "기후변화 대응 위한 국제 협약 체결",
    "전기차 시장 급성장, 내연기관 시대 저물어",
    "메타버스 플랫폼, 교육 분야 진출 본격화",
    "식량 안보 위기, 농업 기술 혁신 시급",
    "우주 탐사 프로젝트, 새로운 행성 발견",
    "반도체 산업, 글로벌 공급망 재편 움직임",
    "청년 실업률 증가, 일자리 대책 필요성 대두",
    "도심 항공 모빌리티, 상용화 단계 진입",
    "사이버 보안 위협 증가, 대응 체계 강화",
    "친환경 에너지 정책, 탄소중립 목표 달성",
    "디지털 화폐 도입, 금융 시스템 변화 예고"
]

# 세션 상태 초기화
if 'current_title' not in st.session_state:
    st.session_state.current_title = None
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'score' not in st.session_state:
    st.session_state.score = None
if 'article_text' not in st.session_state:
    st.session_state.article_text = ""

# 점수 평가 함수
def get_result_message(score):
    """낚시성 확률에 따른 메시지 반환"""
    if score == 100:
        return "🏆 훌륭합니다! 내용이 정확히 일치합니다!", "excellent"
    elif score >= 80:
        return "✨ 좋습니다!  내용이 일치하네요!", "good"
    elif score >= 60:
        return "👍 괜찮습니다. 조금 더 제목에 맞게 작성해보세요!", "average"
    else:
        return "⚠️ 내용이 다릅니다. 내용을 다시 작성해보세요!", "poor"

# 헤더
st.markdown("""
<div class="game-header">
    <h1>📝 나도, 기자? 📝</h1>
    <p style="font-size: 18px; margin-top: 10px;">
        제시된 제목에 맞는 기사를 작성해보세요!<br>
        AI가 당신의 기사가 얼마나 제목에 맞는지 평가합니다.
    </p>
</div>
""", unsafe_allow_html=True)

# 게임 시작 버튼 (게임이 시작되지 않았을 때만 표시)
if not st.session_state.game_started:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🎮 게임 시작하기", type="primary", use_container_width=True):
            st.session_state.current_title = random.choice(NEWS_TITLES)
            st.session_state.game_started = True
            st.session_state.score = None
            st.session_state.article_text = ""
            st.rerun()

# 게임 진행 (점수가 없을 때만 입력 폼 표시)
if st.session_state.game_started and st.session_state.current_title and st.session_state.score is None:
    
    # 제목 표시
    st.markdown(f"""
    <div class="title-card">
        <p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">📰 오늘의 뉴스 제목</p>
        <p class="title-text">{st.session_state.current_title}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 기사 작성 폼
    with st.form(key='article_form'):
        article_text = st.text_area(
            "기사 내용을 작성하세요",
            height=300,
            placeholder="제목에 맞는 내용을 작성해주세요...",
            value=st.session_state.article_text,
            key="article_input"
        )
        
        col1, col2 = st.columns([1, 3])
        submit_button = col1.form_submit_button("✅ 제출하기", type="primary", use_container_width=True)
    
    # 제출 처리
    if submit_button:
        if not article_text or len(article_text.strip()) < 50:
            st.warning("⚠️ 내용을 50자 이상 작성해주세요!")
        else:
            with st.spinner('🤖 AI가 당신의 기사를 분석하고 있습니다...'):
                try:
                    # 낚시성 분석
                    detector = BERTClickbaitDetector(
                        model_path="clickbait_detector_bert2.pt",
                        model_name='klue/bert-base',
                        max_length=256
                    )
                    result = detector.predict(st.session_state.current_title, article_text)
                    clickbait_prob = result['clickbait_probability'] * 100
                    
                    # 점수 계산 (낚시성이 낮을수록 높은 점수)
                    score = max(0, clickbait_prob)
                    
                    st.session_state.score = score
                    st.session_state.article_text = article_text
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 분석 중 오류가 발생했습니다: {e}")
                    st.info("clickbait_detector_bert2.pt 파일이 필요합니다.")

# 결과 표시
if st.session_state.score is not None:
    
    st.markdown("---")
    st.markdown("## 📊 분석 결과")
    
    st.markdown(f"""
    <div class="title-card">
        <p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">제목</p>
        <p class="title-text">{st.session_state.current_title}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="title-card">
        <p style="color: #667eea; font-weight: 600; margin-bottom: 10px;">내용</p>
        <p class="title-text">{st.session_state.article_text}</p>
    </div>
    """, unsafe_allow_html=True)
  
    st.markdown(f"""
    <div class="score-card">
        <p class="score-label">당신의 기사는</p>
        <p class="score-value">{st.session_state.score:.0f}</p>
        <p class="score-label">점입니다</p>
    </div>
    """, unsafe_allow_html=True)
    
    # 평가 메시지
    message, css_class = get_result_message(st.session_state.score)
    st.markdown(f"""
    <div class="result-message {css_class}">
        {message}
    </div>
    """, unsafe_allow_html=True)
    
    # 버튼들
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("🔄 다른 제목으로 도전하기", type="primary", use_container_width=True):
            st.session_state.current_title = random.choice(NEWS_TITLES)
            st.session_state.score = None
            st.session_state.article_text = ""
            st.rerun()
    
    with col2:
        if st.button("🏠 시작화면으로 돌아가기", use_container_width=True):
            st.session_state.current_title = None
            st.session_state.game_started = False
            st.session_state.score = None
            st.session_state.article_text = ""
            st.rerun()

# 게임 설명 (시작 화면일 때만 표시)
if not st.session_state.game_started:
    st.markdown("---")
    st.markdown("### 🎯 게임 방법")
    st.markdown("""
    1. **'게임 시작하기'** 버튼을 클릭합니다
    2. 무작위로 제시되는 **뉴스 제목**을 확인합니다
    3. 제목에 맞는 **기사 내용**을 작성합니다
    4. AI가 당신의 기사를 분석하여 **점수**를 매깁니다
    5. 제목에 맞는 기사를 작성해보세요!
    """)
    
    st.markdown("### 💡 높은 점수를 받는 팁")
    st.markdown("""
    - 제목에 맞는 말투를 사용하세요
    - 제목에 나온 단어를 활용하세요
    - 그럴듯한 내용을 작성하세요
    """)

# 푸터
st.markdown("---")
st.caption("🤖 BERT 기반 낚시성 탐지 AI | ⚡ Streamlit 게임")