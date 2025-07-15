import streamlit as st
st.set_page_config(
    layout="wide",
    page_title="크라우드펀딩 패션 스토리텔링 대시보드",
    page_icon="👗"
)

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import random
import plotly.express as px
import plotly.graph_objects as go
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.font_manager as fm
import streamlit.components.v1 as components
import os


# 폰트 경로를 찾는 함수
def get_font_path():
    # 프로젝트 내 폰트 경로 먼저 확인
    project_font_path = os.path.join(os.path.dirname(__file__), 'fonts', 'NotoSansKR.ttf')
    if os.path.exists(project_font_path):
        return project_font_path
    
    # 시스템 폰트 경로들 확인
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansKR-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/malgun.ttf",  # Windows
    ]
    
    for path in font_paths:
        if os.path.exists(path):
            return path
    
    # 한글 폰트를 찾지 못한 경우 None 반환 (기본 폰트 사용)
    return None


# 한글 폰트 설정
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

# 한글 폰트 설정 시도
try:
    # 시스템에 설치된 한글 폰트 찾기
    font_list = fm.findSystemFonts()
    korean_fonts = []
    for font_path in font_list:
        font_name = fm.FontProperties(fname=font_path).get_name()
        if any(keyword in font_name.lower() for keyword in ['noto', 'nanum', 'malgun', 'gulim', 'dotum']):
            korean_fonts.append(font_name)
    
    if korean_fonts:
        plt.rcParams['font.family'] = korean_fonts[0]
    else:
        # 대체 폰트 설정
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'DejaVu Sans']
except:
    # 폰트 설정 실패 시 기본 설정 유지
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Noto Sans CJK KR', 'NanumGothic', 'Malgun Gothic', 'DejaVu Sans']

# 사이드바 입력 영역 추가
st.sidebar.header("Crowdfunding Fashion Storytelling Dashboard")
category = st.sidebar.selectbox("Category", ["Top", "Jacket", "Jumper", "Padding", "Vest", "Cardigan", "Zip-up", "Coat", "Blouse", "T-shirt", "Knitwear", "Shirt", "Bra top", "Hoodie", "Jeans", "Pants", "Skirt", "Leggings", "Jogger pants", "Dress", "Jumpsuit", "한복"])


season_trend = st.sidebar.selectbox("Season", ["All", "Summer", "Winter", "Spring", "Autumn"])


# 가격대 선택 라디오 버튼 (1개의 열로만 표시)
price_options = [
    "전체",
    "5만원 이하",
    "5만원 ~ 10만원",
    "10만원 ~ 20만원",
    "20만원 ~ 30만원",
    "30만원 이상",
    "직접 입력"
]

price_range_option = st.sidebar.radio(
    "Price Range",
    price_options,
    index=0,
    key="price_radio"
)

# 직접 입력 선택 시 범위 입력
if price_range_option == "직접 입력":
    st.sidebar.markdown("**직접 가격 범위 입력 (단위: 원)**")
    col_min, spacer, col_max = st.sidebar.columns([1, 0.1, 1])
    with col_min:
        custom_min = st.sidebar.text_input("시작 가격", value="10000", key="custom_min")
    with spacer:
        st.sidebar.markdown("<div style='text-align:center; font-size:20px; padding-top: 30px;'>~</div>", unsafe_allow_html=True)
    with col_max:
        custom_max = st.sidebar.text_input("끝 가격", value="500000", key="custom_max")
    try:
        price_min = int(custom_min.replace(",", ""))
        price_max = int(custom_max.replace(",", ""))
    except ValueError:
        st.sidebar.error("숫자를 올바르게 입력해주세요.")
        price_min, price_max = 10000, 500000
    price_range = (price_min, price_max)
else:
    # 각 구간별 실제 범위 할당
    if price_range_option == "전체":
        price_range = (10000, 500000)
    elif price_range_option == "5만원 이하":
        price_range = (10000, 50000)
    elif price_range_option == "5만원 ~ 10만원":
        price_range = (50000, 100000)
    elif price_range_option == "10만원 ~ 20만원":
        price_range = (100000, 200000)
    elif price_range_option == "20만원 ~ 30만원":
        price_range = (200000, 300000)
    elif price_range_option == "30만원 이상":
        price_range = (300000, 5000000)
    else: # 예외 처리
        price_range = (10000, 500000)

def get_element_order(df):
    element_orders = df.groupby("campaign_id")["element"].apply(list)
    order_counts = Counter(tuple(order) for order in element_orders)
    return list(order_counts.most_common(1)[0][0]) if order_counts else []

def get_keywords(df, element):
    sentences = df[df["element"] == element]["sentence"]
    words = " ".join(sentences).replace("\n", " ").split()
    words = [w.strip(".,!\"'()[]") for w in words if len(w) > 1]
    return words

def get_example_sentences(df, element, n=3):
    examples = df[df["element"] == element]["sentence"].unique()
    return random.sample(list(examples), min(n, len(examples)))

def get_top_bigrams(df, element, top_n_words=5, top_n_bigrams=3):
    sentences = df[df["element"] == element]["sentence"]
    text = " ".join(sentences).replace("\n", " ")

    vectorizer = CountVectorizer(ngram_range=(2, 2))
    X = vectorizer.fit_transform([text])
    bigram_counts = X.toarray().sum(axis=0)
    bigram_vocab = vectorizer.get_feature_names_out()
    bigram_freq = dict(zip(bigram_vocab, bigram_counts))

    words = get_keywords(df, element)
    word_counter = Counter(words)
    top_words = [w for w, _ in word_counter.most_common(top_n_words)]

    result = {}
    for word in top_words:
        related_bigrams = [bg for bg in bigram_freq if word in bg.split()]
        sorted_bigrams = sorted(related_bigrams, key=lambda x: bigram_freq[x], reverse=True)
        result[word] = sorted_bigrams[:top_n_bigrams]
    return result

# 📌 CSS 스타일 정의
st.markdown("""
    <style>
    .description-box {
        background-color: #F9F9F9;
        border-left: 5px solid #B0E0E6;
        padding: 20px;
        margin-bottom: 20px;
        border-radius: 10px;
        font-size: 15px;
        line-height: 1.6;
    }
    .description-title {
        font-size: 36px;
        font-weight: bold;
        color: ##B0E0E6;
        margin-top: 10px;
        margin-bottom: 12px;
    }
    </style>
""", unsafe_allow_html=True)

# 🎯 제목 (박스 바깥에 따로 출력)
st.markdown('''<div class="description-title">크라우드펀딩 패션 스토리텔링 분석 대시보드</div>''', unsafe_allow_html=True)

# 📘 설명 박스
st.markdown("""
<div class="description-box">
    <strong>성공적으로 종료된 패션 크라우드펀딩 캠페인</strong>의 스토리텔링 전략을 분석하여  
    효과적인 스토리 작성을 위한 핵심 요소별 키워드와 실용적인 정보를 제공합니다.<br>
    사용자는 <strong>카테고리 / 시즌 / 가격대</strong>별 필터링을 통해 다음과 같은 맞춤형 분석 결과를 확인할 수 있습니다.
    <ul style="margin-top: 8px; margin-left: 20px; line-height: 1.7;">
        <li>스토리 구성 순서</li>
        <li>전체 키워드 분석</li>
        <li>핵심 요소별 주요 키워드 & 예시 문장</li>
    </ul>
    <div style="margin-top:10px; font-size: 14px; color: #555;">
        💡 <strong>Tip:</strong> 각 요소별 키워드 및 예시 문장, 그리고 스토리 구성 순서를 활용하여  
        타겟 고객에게 가장 효과적으로 어필할 수 있는 스토리텔링 전략을 수립해보세요!
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.markdown("## 스토리 구성 순서")

# 📊 구성 요소 순서 안내
st.markdown("""
<div class="description-box">
    문제 제기 및 솔루션 제시 → 제품 전달 가치 → 제품 상세 설명 → 브랜드 소개 →  
    제품 및 브랜드 외부 평가 → 펀딩 참여 유도 → 자주 묻는 질문
</div>
""", unsafe_allow_html=True)




st.markdown("""
<style>
/* 전체 배경을 흰색으로, 기본 글씨는 어두운 회색 */
body, .stApp {
    background-color: white !important;
    color: #222222 !important;
    font-family: 'Helvetica', sans-serif;
}

/* 박스 스타일: 연회색 배경, 둥근 테두리 */
.custom-box {
    border: 1px solid #DDD;
    border-radius: 10px;
    padding: 20px;
    background-color: #F9F9F9;
    margin-bottom: 20px;
    color: #333;
}
.custom-box h4 {
    margin-top: 0;
    font-size: 18px;
    font-weight: 600;
    color: #3C77FF;  /* 파스텔 블루 */
}

/* 표(table) 스타일: 밝은 테마용 */
table {
    border-collapse: collapse;
    width: 100%;
}
th, td {
    border: 1px solid #CCC;
    padding: 6px 10px;
    text-align: left;
    font-size: 14px;
    color: #222;
}

/* 헤더 컬럼: 연한 파란 배경 + 선명한 컬러 */
th {
    background-color: #E5F1FF;
    color: #3366CC;
}

/* 탭 텍스트 진하게 */
.stTabs [data-baseweb="tab"] {
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)


# 키워드 및 예시 데이터
functional_keywords = {
    "실용성": "복잡한세상 편하게 살자! 실용성, 효율성 없으면 말짱 꽝이죠.",
    "고정력": "목둘레가 쉽게 늘어나지 않도록 헤리 테이프를 추가 봉제하여 탄탄하게 고정.",
    "휴대": "가방 한켠에 쏘옥-, 주머니에 쏘옥- 간편한 휴대성!",
    "디테일": "사소한 디테일까지 고심하여 제작하였습니다.",
    "착용감": "입은 듯 안 입은 듯 착용감이 뛰어납니다.",
    "구김": "구김이 적어 관리가 쉽습니다.",
    "사이즈": "사이즈 선택 고민을 줄이는 가이드 제공!",
    "내구성": "프리미엄 소재와 내구성 있는 자재로 제작.",
    "편안함": "무엇 하나 거슬리지 않는 편안함을 위해 연구.",
    "효율성": "효율성을 위해 작은 구조까지 개선.",
    "원단": "톡톡한 두께감의 원단으로 선택."
}

emotional_keywords = {
    "자유로움": "자유로움과 개성을 느낄 수 있게 해줍니다.",
    "나만의": "나만의 공간과 감성을 담아낸 디자인.",
    "자존심": "10년간의 자존심을 걸고 만든 맨투맨.",
    "첫인상": "좋은 첫인상을 위한 포인트 아이템.",
    "색다른 느낌": "색다른 느낌을 주고 싶은 날 스타일링하기 좋습니다.",
    "섹시함": "커프스로 섹시함 3cm 더 키워보세요!",
    "고급스러움": "고급스러움을 자랑하는 디테일.",
    "트렌디함": "트렌디하고 스마트한 무드를 담았습니다.",
    "디자인": "캐주얼한 디자인과 클래식함의 조화."
}

# 도넛 차트용 값
emotional_ratio = 32.7
functional_ratio = 67.3
labels = ['감성적 키워드', '기능적 키워드']
sizes = [emotional_ratio, functional_ratio]
colors = ['#ffb3c6', '#b3d9ff']

# 🔷 공통 스타일 정의
box_style = """
<style>
.keyword-box {
    display: inline-block;
    padding: 8px 16px;
    margin: 6px;
    border-radius: 12px;
    border: 1px solid #ccc;
    background-color: #f9f9f9;
    transition: all 0.3s ease;
    cursor: pointer;
    font-size: 15px;
}
.keyword-box:hover {
    background-color: #e0f0ff;
    border-color: #4099ff;
    color: #004080;
}
#tooltip {
    margin-top: 20px;
    padding: 12px;
    background-color: #f0f8ff;
    border: 1px dashed #4099ff;
    border-radius: 10px;
    font-size: 14px;
    min-height: 50px;
}
</style>
"""

def render_hover_box(title, keywords_dict):
    html = f"""
    {box_style}
    <h4>{title}</h4>
    <div>
    """
    for kw, ex in keywords_dict.items():
        html += f"""<span class="keyword-box" onmouseover="document.getElementById('tooltip').innerText='{ex}'">{kw}</span>"""
    html += "</div><div id='tooltip'></div>"
    components.html(html, height=300, scrolling=False)

# # 🔷 대시보드 출력
# st.markdown("## 💭 감성 vs 기능적 키워드 분석")

# # 🔸 레이아웃: 도넛 차트 + 키워드 hover 탭
# col1, col2 = st.columns([1, 2])

# with col1:
#     st.markdown("### 🧁 감성 vs 기능 도넛 차트")
#     fig, ax = plt.subplots(figsize=(4, 4))
#     wedges, texts, autotexts = ax.pie(
#         sizes, labels=labels, colors=colors,
#         autopct='%1.1f%%', startangle=90, counterclock=False,
#         wedgeprops=dict(width=0.6)
#     )
#     ax.axis('equal')
#     # ✅ 범례 추가
#     ax.legend(
#         wedges,
#         labels,
#         title="범례",
#         loc="center left",
#         bbox_to_anchor=(1, 0, 0.5, 1),
#         labelcolor="black",
#         facecolor="white",
#         edgecolor="gray"
#     )
#     st.pyplot(fig)

# with col2:
#     tabs = st.tabs(["⚙️ 기능적 키워드", "💖 감성적 키워드"])
#     with tabs[0]:
#         render_hover_box("기능적 키워드", functional_keywords)
#     with tabs[1]:
#         render_hover_box("감성적 키워드", emotional_keywords)

st.markdown("---")

# 감성 vs 기능 도넛 차트 레이아웃 (plotly + 오른쪽 탭)
def render_emotion_function_donut_chart():
    st.markdown("## 전체 키워드 분석")

    left, right = st.columns([1.1, 1.9])

    with left:
        pastel_colors = ["#FFB3C6", "#B3D9FF"]

        fig = px.pie(
            names=labels,
            values=sizes,
            hole=0.4,
            color_discrete_sequence=pastel_colors
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            paper_bgcolor="white",
            plot_bgcolor="white",
            font_color="black",
            legend=dict(font=dict(color="black"))
        )
        fig.update_traces(
            textinfo='percent',
            textfont_size=14,
            textfont_color='black'
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        tabs = st.tabs(["기능적 키워드", "감성적 키워드"])
        with tabs[0]:
            render_hover_box("기능적 키워드", functional_keywords)
        with tabs[1]:
            render_hover_box("감성적 키워드", emotional_keywords)

render_emotion_function_donut_chart()



st.markdown("---")
st.markdown("## 핵심 요소별 주요 키워드 & 예시 문장")

problem_keywords = {
    "불편함": 60,
    "세탁 용이": 55,
    "맞춤형 추천": 50,
    "합리적 가격": 48,
    "착용감": 45,
    "스타일 다양성": 43,
    "높은 비용": 40,
    "다양한 사이즈": 36,
    "체형 보완": 35,
    "TPO": 33,
    "가성비": 31,
    "계절감": 28,
    "친환경": 15,
    "코디 고민": 24,
    "디테일 강조": 22,
    "고급 원단": 15,
    "기능 제한": 12,
    "기능성 원단" : 18,
    "활용도" : 20
}


# 솔루션 제시 키워드와 빈도
# solution_keywords = {
#     "간편함": 60,
#     "효율성": 55,
#     "혁신적 방법": 50,
#     "빠른 해결": 48,
#     "사용자 친화적": 45,
#     "접근 용이성": 43,
#     "자동화": 40,
#     "모바일 최적화": 38,
#     "UI/UX 개선": 36,
#     "시간 절약": 35,
#     "저비용 솔루션": 33,
#     "직관적 인터페이스": 31,
#     "원클릭 기능": 28,
#     "데이터 기반": 26,
#     "맞춤형 제공": 24,
#     "설치 간편": 22,
#     "고객 중심 설계": 20,
#     "문제 자동 진단": 18,
#     "학습 필요 없음": 15,
#     "실시간 피드백": 12
# }


element_example_sentences = {
    # 파이차트용 요소 (세부 분류 O)
    "브랜드 소개": {
        "브랜드 가치관": [
            "우리는 옷을 통해 삶의 가치를 높인다는 철학을 가지고 있습니다.",
            "모두가 입을 수 있고 소중한 삶의 보탬이 되어줄 제품을 만드는 것이 유니핏의 신념이자 추구하는 방향입니다."
        ],
        "창작자의 스토리": [
            "학창시절부터 패션을 사랑해온 디자이너의 열정이 담겼습니다.",
            "직접 겪은 실패와 회복의 경험이 이 프로젝트의 출발점이었습니다."
        ],
        "실패 극복": [
            "초기 제작 실패를 딛고 수차례 개선을 거쳐 완성했습니다.",
            "고객 피드백을 반영하여 핏과 소재를 전면 수정했습니다."
        ]
    },

    "제품 및 브랜드 외부 평가": {
        "사용자 후기": [
            "1,500개 이상의 구매 후기에서 4.9점의 평점을 기록했습니다.",
            "후기 대부분이 '핏이 좋다', '재질이 고급스럽다'는 반응입니다."
        ],
        "기관 인증": [
            "KC 인증과 함께 OEKO-TEX 친환경 인증을 획득했습니다.",
            "안심하고 착용하실 수 있도록 국가 품질 인증을 완료했습니다.",
            "철저한 안전관리 및 프로세스 검침은 기본! 거기에 더해, 기본적으로행복한 직원이 훌륭한 제품을 만든다고 생각하기 때문에 받은WRAP 인증까지!"
        ],
        "수상 경력": [
            "국제 섬유 디자인 대회인 IFDA 2022에서 본 제품의 원단 배색과 패턴 디자인이 심사위원 만장일치로 우수상을 수상했습니다.",
            "소비자가 뽑은 브랜드 대상 2년 연속 수상’은 저희 제품을 직접 경험하신 수많은 고객분들의 평가 덕분이었습니다.",
            "저희 브랜드는 2023 K패션 어워즈에서 ‘올해의 혁신 디자인’ 부문을 수상하며 제품력과 디자인 모두를 인정받았습니다."
        ]
    },

    "펀딩 참여 유도": {
        "한정 수량": [
            "얼리버드 한정 수량으로 20% 할인 혜택을 드립니다.",
            "재고 소진 시 추가 구매가 불가합니다."
        ],
        "얼리버드": [
            "펀딩 초반 참여자에게만 제공되는 스페셜 리워드입니다.",
            "48시간 이내 얼리버드 참여자에겐 특별 패키지를 드립니다."
        ],
        "스페셜 리워드": [
            "목표 금액 달성 시 추가 리워드를 드립니다.",
            "펀딩 참여자 전용 한정판 굿즈를 제공합니다."
        ]
    },

    "자주 묻는 질문": {
        "배송일정": [
            "리워드 수령으로부터 14일 이내에 발생한 초기 하자에 대해서는 본 A/S정책이 적용되지 않습니다.",
            "세탁, 사용, 택 제거, 오염, 수선 등 이후 발생한 문제는 유상수리 및 왕복 택배비는 서포터님 부담으로 진행되며, 경우에 따라 (유상 수리가 불가할 정도로 심각한 훼손의 경우) 수리가 불가할 수 있습니다.",
            "단순 변심에 의한 환불 및 교환은 불가합니다.(해당 상품은 네팔에서 직접 만들어서 오는 펀딩 상품으로 반품이나 교환이 쉽지 않습니다. 상품 자체의 문제일 경우 환불이 가능하지만, 그렇지 않은 경우 환불 및 교환이 불가함을 미리 공지드립니다."
        ],
        "세탁방법": [
            "Q. 세탁 방법은 어떻게 되나요? 상세 페이지 하단에 상세한 세탁 방법을 안내드립니다. 드라이클리닝을 권장하며, 손세탁일 경우, 미온수와 중성세제를 사용하여 가볍게 세탁하시고 그늘진 곳에서 자연건조 하시기 바랍니다.",
            "Q. 세탁시 주의사항이 있을까요? A. 세탁기에 30도 이하로 세탁하시면 됩니다. 면은 뜨거운 열을 가하면 줄어드는 게 필연이라, 건조기에는 절대 돌리지 마세요~ 축률을 최소화한 공정을 거쳤기 때문에 찬물로 빠시면 3% 이내로 축률을 막을 수 있습니다. 세탁기에서 꺼내신 후 널어서 말리시는 게 제일 좋습니다.",
            "Q. 세탁은 어떻게 하나요?A. 세탁은 드라이 크리닝 하시길 권장합니다."
        ],
        "교환/반품": [
            "펀딩 마감 이후, 불가피한 사유로 배송지 변경이 필요하시다면 해당 페이지 내 '메이커에게 문의하기'를 통해 문의 부탁드립니다.",
            "배송은 언제 시작되나요? A. 결제는 펀딩 기간이 종료 된 후 다음날부터 4일 동안 진행이 됩니다.",
            "펀딩 기간 종료와 동시에 배송이 시작됩니다. 일반 배송은 4일동안 진행되며, 제주/도서산간 지역 배송은 최대 7일이 걸릴 수 있습니다."
        ]
    },

    # 워드클라우드용 요소 (단일 리스트)
    "문제 제기 및 솔루션 제시": [
        # "캐시미어는 정말 좋은 소재지만 좋은 소재에 가려져 원단만 강조되고 알게 모르게 옷이라면 지녀야 할 편안함, 디자인성들이 뒷전으로 가있는 소재이기도 합니다.",
        # "기존의 점퍼 공식인, 30데니아 - 3레이어로도 샘플 테스트를 해봤는데, 역시 예상대로 리버서블에겐 너무 두껍고, 거슬리고, 움직임마저 편하지 않았어요.",
        "여름 티셔츠는 비침이 심하거나 땀이 배어 불편합니다. 특수 가공 원단으로 땀 배임 없이 쾌적함을 유지합니다.",
        "핏이 어정쩡하거나, 세탁 후 변형이 심한 옷이 많습니다. 세탁 후에도 형태 유지력이 뛰어난 소재를 사용했습니다.",
        "매번 어울리는 옷 찾기가 어려워 스트레스를 받습니다. 베이직하면서도 고급스러운 핏으로 어떤 상황에서도 활용도 높습니다.",
    ],

    "제품 상세 설명": [
        "핏은 레귤러 핏으로, 슬림하지도 벙벙하지도 않아 누구에게나 잘 어울립니다.",
        "소재는 100% 코튼이며, 피부에 자극 없이 부드럽게 닿습니다.",
        "컬러는 블랙, 아이보리, 그레이 등 데일리로 활용하기 좋습니다."
    ],

    "제품 전달 가치": [
        "기능적(F): 복잡한세상편하게살자! 바쁘고 바쁜 우리네 삶 실용성, 효율성 없으면 말짱 꽝이죠. [단정함]을 필요로 할 때 입을 수 있도록 휴대하기 쉽게! 가방 한켠에 쏘옥-, 주머니에 쏘옥- 정신없고답 없는 상황에서해답은 셔츠토시 뿐!",
        "표현적(E): 도시적인 세련미를 표현할 수 있습니다.",
        "심미적(A): 트렌드에 얽메이지 않는 유니크한 디자인 꽈배기, 와플, 베이직 니트 등등 베이직 디자인으로 식상했다면 유니크의 차별화된 디자인으로 매년 유니크하게 연출할 수 있습니다."
    ]
}


element_analysis_info = [
    {"name": "브랜드 소개", "method": "세부 카테고리 요소 추출", "examples": ["브랜드 가치관", "창작자의 스토리", "실패 극복"], "chart_type": "pie"},
    {"name": "문제 제기 및 솔루션 제시", "method": "키워드 빈도 분석", "examples": problem_keywords, "chart_type": "wordcloud"},
    {"name": "제품 상세 설명", "method": "TTA 기반 키워드 분석", "examples": ["소재: 면", "핏: 루즈", "컬러: 블랙"], "chart_type": "treemap"},
    {"name": "제품 전달 가치", "method": "FEA 기반 추출", "examples": ["Functional: 편안함", "Expressive: 감각적", "Aesthetic: 디자인"], "chart_type": "radar"},
    {"name": "제품 및 브랜드 외부 평가", "method": "세부 요소 추출", "examples": ["사용자 후기", "기관 인증", "수상 경력"], "chart_type": "pie"},
    {"name": "펀딩 참여 유도", "method": "세부 요소 추출", "examples": ["한정 수량", "얼리버드", "스페셜 리워드"], "chart_type": "pie"},
    {"name": "자주 묻는 질문", "method": "세부 요소 추출", "examples": ["배송일정", "세탁방법", "교환/반품"], "chart_type": "pie"}
]

# # 1. 도넛형 파이차트 + 병합형 테이블 (브랜드 소개, 외부 평가 등)
# def render_pie_chart(title, labels):
#     st.markdown(f"### 🔸 {title}")
#     left, right = st.columns([1.1, 1.9])

#     with left:
#         values = [random.randint(10, 30) for _ in labels]

#         neon_theme_colors = [
#             "#FF6EC7", "#72F0FF", "#C586FF", "#E68CFF", "#8AD6FF", "#C9A8FF"
#         ]
#         color_seq = neon_theme_colors[:len(labels)]

#         fig = px.pie(
#             names=labels,
#             values=values,
#             hole=0.4,
#             color_discrete_sequence=color_seq
#         )
#         fig.update_layout(
#             margin=dict(l=10, r=10, t=10, b=10),
#             height=300,
#             paper_bgcolor="#0D0C2B",
#             plot_bgcolor="#0D0C2B",
#             font_color="white",
#             legend=dict(font=dict(color="white"))
#         )
#         fig.update_traces(
#             textinfo='percent',
#             textfont_size=14,
#             textfont_color='white'
#         )
#         st.plotly_chart(fig, use_container_width=True)

#     # 오른쪽: element_example_sentences에서 문장 출력
#     with right:
#         st.markdown("### 💬 세부 요소별 문장 예시")

#         example_data = element_example_sentences.get(title, {})

#         if isinstance(example_data, dict):
#             for sub_elem, sentences in example_data.items():
#                 with st.expander(f"📌 {sub_elem}"):
#                     for s in sentences:
#                         st.markdown(f"- {s}")
#         else:
#             st.info("예시 문장이 없습니다.")

def render_pie_chart(title, labels):
    st.markdown(f"### {title}")
    left, right = st.columns([1.1, 1.9])

    with left:
        values = [random.randint(10, 30) for _ in labels]

        # 🎨 파스텔/네온 컬러 (필요시 바꿔도 OK)
        pastel_colors = [
            "#FFB3BA", "#FFDAC1", "#E2F0CB", "#C9C9FF", "#B5EAD7", "#D5AAFF"
        ]
        color_seq = pastel_colors[:len(labels)]

        fig = px.pie(
            names=labels,
            values=values,
            hole=0.4,
            color_discrete_sequence=color_seq
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=10, b=10),
            height=300,
            paper_bgcolor="white",     # 전체 배경 흰색
            plot_bgcolor="white",
            font_color="black",         # 텍스트 색상
            legend=dict(font=dict(color="black"))
        )
        fig.update_traces(
            textinfo='percent',
            textfont_size=14,
            textfont_color='black'     # 퍼센트 텍스트 색상 🔽 여기!
        )
        st.plotly_chart(fig, use_container_width=True)

    # 오른쪽: 예시 문장
    with right:
        # st.markdown("💬 세부 요소별 문장 예시")
        st.markdown(
        "<h4 style='text-align: center; font-weight: bold;'>세부 요소별 문장 예시</h4>",
        unsafe_allow_html=True
    )

        example_data = element_example_sentences.get(title, {})

        if isinstance(example_data, dict):
            for sub_elem, sentences in example_data.items():
                with st.expander(f"{sub_elem}"):
                    for s in sentences:
                        st.markdown(f"- {s}")
        else:
            st.info("예시 문장이 없습니다.")

# 사용자 정의 진한 색상 팔레트
custom_colors = ["#6D9FB3", "#B1CBA1", "#F0BA89", "#E89A9A", "#E36C75"]

# 컬러 펑션 정의
def multicolor_func(*args, **kwargs):
    return random.choice(custom_colors)

# ✅ 함수 정의
def render_wordcloud(title: str, keyword_freq: dict, problem_example_sentences: list):
    # 워드클라우드 객체 생성
    font_path = get_font_path()
    
    try:
        if font_path:
            wc = WordCloud(
                font_path=font_path,
                background_color="white",
                width=400,
                color_func=multicolor_func,
                height=300,
                max_font_size=40,
                min_font_size=10
            ).generate_from_frequencies(keyword_freq)
        else:
            # 폰트 경로 없이 기본 설정으로 생성
            wc = WordCloud(
                background_color="white",
                width=400,
                color_func=multicolor_func,
                height=300,
                max_font_size=40,
                min_font_size=10
            ).generate_from_frequencies(keyword_freq)
    except Exception as e:
        # 폰트 로드 실패 시 기본 설정으로 재시도
        st.warning(f"폰트 로드 중 오류 발생: {str(e)[:100]}... 기본 폰트를 사용합니다.")
        wc = WordCloud(
            background_color="white",
            width=400,
            color_func=multicolor_func,
            height=300,
            max_font_size=40,
            min_font_size=10
        ).generate_from_frequencies(keyword_freq)

    # 워드클라우드 시각화
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.patch.set_facecolor('white')

    # 레이아웃: 왼쪽 워드클라우드 / 오른쪽 문장
    st.markdown(f"### {title}")
    left, right = st.columns([1.2, 1.8])

    with left:
        st.pyplot(fig)

    with right:
        # st.markdown("**예시 문장:**")
        st.markdown(
        "<h4 style='text-align: center; font-weight: bold;'>문장 예시</h4>",
        unsafe_allow_html=True)
        for sentence in problem_example_sentences:
            st.markdown(f"- {sentence}")


# def render_treemap():
#     df = pd.DataFrame({
#         "category": ["style", "style", "fit", "fit", "material", "material", "color", "color"],
#         "keyword": ["캐주얼", "페미닌", "루즈", "슬림", "면", "폴리", "블랙", "크림"],
#         "count": [30, 20, 15, 25, 35, 18, 28, 22]
#     })

#     # 📘 컬러맵: 은은한 블루 민트 계열
#     fig = px.treemap(
#         df,
#         path=['category', 'keyword'],
#         values='count',
#         color='count',
#         color_continuous_scale=[
#             "#A9BCD0", "#778DA9", "#415A77", "#1B263B"
#         ],
#     )

#     # 🌙 어두운 배경 및 폰트 컬러 조정
#     fig.update_layout(
#         paper_bgcolor="#0D1B2A",   # 전체 배경
#         plot_bgcolor="#0D1B2A",
#         font=dict(color="#E0E1DD"),
#         margin=dict(t=25, l=0, r=0, b=0)
#     )

#     st.plotly_chart(fig, use_container_width=True)

# def render_treemap():
#     df = pd.DataFrame({
#         "category": [
#             "핏(fit)", "핏(fit)", "핏(fit)", "핏(fit)",
#             "색상(hue)", "색상(hue)", "색상(hue)",
#             "원단 종류(material)", "원단 종류(material)", "원단 종류(material)",
#             "스타일(style)", "스타일(style)", "스타일(style)"
#         ],
#         "type": [
#             "팬츠(pants)", "스커트(dress)", "아우터(top)", "아우터(top)",
#             "B계열", "RP계열", "R계열",
#             "천연 소재", "합성 소재", "재생소재",
#             "모던(modern)", "페미닌(feminine)", "스포티(sporty)"
#         ],
#         "keyword": [
#             "레귤러", "A라인", "타이트", "오버사이즈",
#             "블루", "라벤더", "레드",
#             "코튼", "폴리에스터", "레이온",
#             "미니멀", "로맨틱", "캐주얼"
#         ],
#         "count": [
#             35, 25, 15, 20,
#             30, 18, 15,
#             28, 22, 10,
#             18, 21, 27
#         ]
#     })

#     fig = px.treemap(
#         df,
#         path=['category', 'type', 'keyword'],
#         values='count',
#         color='count',
#         color_continuous_scale=[
#             "#FF6EC7",  # 네온 핑크
#             "#C586FF",  # 연보라
#             "#72F0FF",  # 네온 블루
#             "#E68CFF",  # 연보라핑크
#             "#8AD6FF",  # 민트블루
#             "#B388EB",  # 라이트 퍼플
#             "#FCD3DE",  # 밝은 핑크
#             "#D5AAFF",  # 연한 라벤더
#         ]
#     )

#     fig.update_layout(
#         paper_bgcolor="white",   # 전체 배경색
#         plot_bgcolor="white",    # 내부 배경색
#         font=dict(color="black"),  # 흰색 폰트
#         margin=dict(t=25, l=0, r=0, b=0)
#     )

#     # ⬅️ 노드 테두리 선 색도 흰 테마에 맞게 명시
#     fig.update_traces(
#         marker=dict(line=dict(color='white', width=1))
#     )

#     st.plotly_chart(fig, use_container_width=True)


def render_treemap():
    st.markdown("""
    <h3 style='margin-bottom: -5px;'>제품 상세 설명</h3>
    <style>
    /* plotly 차트를 포함하는 div의 상단 마진 제거 */
    .element-container:has(.js-plotly-plot) {
        margin-top: 0px !important;
        padding-top: 0px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    df = pd.DataFrame({
        "category": [
            "핏(fit)", "핏(fit)", "핏(fit)", "핏(fit)",
            "색상(hue)", "색상(hue)", "색상(hue)",
            "원단 종류(material)", "원단 종류(material)", "원단 종류(material)",
            "스타일(style)", "스타일(style)", "스타일(style)"
        ],
        "type": [
            "베스트(vest)","티셔츠(tee)", "셔츠(shirt)", "셔츠(shirt)",
            "B계열", "RP계열", "R계열",
            "천연 소재", "합성 소재", "재생소재",
            "모던(modern)", "페미닌(feminine)", "스포티(sporty)"
        ],
        "keyword": [
            "레귤러","레귤러", "타이트", "오버사이즈",
            "블루", "라벤더", "레드",
            "코튼", "폴리에스터", "레이온",
            "미니멀", "로맨틱", "캐주얼"
        ],
        "count": [
            35, 25, 15, 20,
            30, 18, 15,
            28, 22, 10,
            18, 21, 27
        ]
    })
    df["root"] = " "
    
    # 호버 시 표시할 추가 정보
    df["percentage"] = (df["count"] / df["count"].sum() * 100).round(1)
    df["description"] = [
        "편안한 일상 착용감", "우아한 실루엣", "몸에 맞는 핏", "여유로운 착용감",
        "시원하고 차분한 느낌", "로맨틱하고 부드러운 색감", "열정적이고 강렬한 인상",
        "자연스럽고 친환경적", "내구성이 뛰어남", "지속가능한 소재",
        "깔끔하고 세련된 스타일", "우아하고 여성스러운 분위기", "활동적이고 편안한 룩"
    ]
    
    # 각 키워드별 예시 문장 추가
    df["example_sentence"] = [
        "몸에 무리가 없는 레귤러 핏으로 편안한 착용감을 제공합니다.",
        "여성스러운 A라인 실루엣으로 우아한 분위기를 연출해요.",
        "슬림한 타이트 핏으로 몸매가 돋보이는 스타일링이 가능합니다.",
        "넉넉한 오버사이즈로 트렌디하고 편안한 룩을 완성할 수 있어요.",
        "차분하고 시원한 블루 컬러로 깔끔한 코디가 가능합니다.",
        "로맨틱한 라벤더 색상으로 부드러운 매력을 표현해보세요.",
        "강렬한 레드 컬러로 포인트를 주어 시선을 사로잡습니다.",
        "100% 순면 코튼으로 부드럽고 통기성이 뛰어납니다.",
        "폴리에스터 소재로 내구성이 좋고 관리가 간편해요.",
        "부드러운 레이온 소재로 실키한 터치감이 특징입니다.",
        "미니멀한 디자인으로 어떤 스타일링에도 잘 어울려요.",
        "로맨틱한 디테일로 여성스러운 무드를 완성합니다.",
        "캐주얼한 스타일로 데일리 룩에 완벽한 아이템이에요."
    ]

    fig = px.treemap(
        df,
        path=['root','category', 'type', 'keyword'],
        values='count',
        color='count',
        color_continuous_scale=[
            "#FFF0F5", "#FFD1DC", "#FFECB3",
            "#D1F2EB", "#D6EAF8", "#E8DAEF",
            "#FADBD8", "#FDEDEC"
        ],
        template="plotly_white",
        # 호버 시 표시할 추가 데이터
        custom_data=['percentage', 'description', 'example_sentence']
    )   

    fig.update_traces(
        root_color="white",
        marker=dict(
            colorscale=None,
            line=dict(color="white", width=2)  # 기본 테두리는 얇은 회색
        ),
        selector=dict(type='treemap'),
        # 호버 템플릿 커스터마이징 (예시 문장만 표시)
        hovertemplate="""<b>%{label}</b><br>- 예시 문장: %{customdata[2]}<extra></extra>"""
    )

    # 호버 박스 스타일 조정
    fig.update_layout(
        margin=dict(t=0, l=0, r=0, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=dict(color="black"),
        treemapcolorway=[
            "#FFFFFF",  # 루트용 흰색
            "#FFD1DC",  # 파스텔 핑크
            "#AEC6CF",  # 파스텔 블루
            "#FFFACD",  # 파스텔 옐로우
            "#BFD8B8",  # 파스텔 민트
            "#E0BBE4",  # 라일락
            "#FFB347",  # 피치 오렌지
            "#B2EBF2",  # 밝은 아쿠아
            "#F5CBA7"   # 크림 베이지
        ],
        hoverlabel=dict(
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="gray",
            font_size=12,
            font_family="Arial",
            align="left"
        )
    )
    
    # CSS를 추가하여 hover 시 테두리 효과 적용
    st.markdown("""
    <style>
    .js-plotly-plot .plotly .treemap-trace path:hover {
        stroke: white !important;
        stroke-width: 4px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    st.plotly_chart(fig, use_container_width=True, theme=None)


# def render_radar_chart():
#     categories = ['Functional', 'Expressive', 'Aesthetic']
#     values = [random.randint(20, 40) for _ in categories]
#     fig = go.Figure()
#     fig.add_trace(go.Scatterpolar(
#         r=values + [values[0]],
#         theta=categories + [categories[0]],
#         fill='toself',
#         line_color="#72F0FF"
#     ))
#     fig.update_layout(
#         polar=dict(bgcolor="white"),
#         showlegend=False,
#         height=350
#     )
#     st.plotly_chart(fig, use_container_width=True)

def render_radar_chart():
    st.markdown("### 제품 전달 가치")

    col1, col2 = st.columns([1.2, 1.8])

    with col1:
        categories = ['Functional', 'Expressive', 'Aesthetic']
        values = [35, 45, 25]  # 예시 점수

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=categories + [categories[0]],
            fill='toself',
            line=dict(color='rgba(255, 99, 132, 1)', width=3),
            fillcolor='rgba(255, 99, 132, 0.3)',
            marker=dict(size=8)
        ))

        fig.update_layout(
            polar=dict(
                bgcolor="white",
                radialaxis=dict(
                    visible=True,
                    range=[0, 50]
                ),
                angularaxis=dict(
                    tickfont=dict(size=13),  # 글씨 작게 해서 잘림 방지
                    rotation=90,             # 시작 위치 변경 (옵션)
                    direction="clockwise"
                )
            ),
            showlegend=False,
            height=420,
            width=420,
            margin=dict(l=40, r=40, t=40, b=40)
        )

        st.plotly_chart(fig, use_container_width=False)

    with col2:
        # st.markdown("### 💬 예시 문장")
        st.markdown(
        "<h4 style='text-align: center; font-weight: bold;'>문장 예시</h4>",
        unsafe_allow_html=True)

        with st.expander("Functional", expanded=False):
            st.markdown("- 복잡한 세상 편하게 살자! ... 셔츠토시 뿐!")
            st.markdown("- 몸에 닿았을 때의 편안함과 부드러움을 ... 신경쓰이는 곳이 없어야 하기 때문이에요.")
            st.markdown("- 다리 밑단이 말려 올라가는 문제를 해결하기 위해 ... 선을 이동했습니다.")

        with st.expander("Expressive", expanded=False):
            st.markdown("- 뽐내며 모든 이들에게 영향력을 주고 ... 영감을 받는다.")
            st.markdown("- KEEP IT REAL 너답게 해! 있는 그대로의 너도 괜찮아.")
            st.markdown("- 내 감정을 마음껏~ 표현할 수 있어요 ... 나를 더 잘 알게 될 거예요.")

        with st.expander("Aesthetic", expanded=False):
            st.markdown("- 트렌디한 와이드 카라 XXHIT 셔츠토시 ... 스타일링하기 좋습니다.")
            st.markdown("- 트렌드에 얽메이지 않는 유니크한 디자인 ... 유니크하게 연출할 수 있습니다.")
            st.markdown("- 티셔츠 자체의 핏을 흐리지 않는 얇지않고 ... 결국 그러한 원단을 찾았습니다.")

# 출력
# for info in element_analysis_info:
#     with st.expander(f"🔸 {info['name']}"):
#         if info["chart_type"] == "pie":
#             render_pie_chart(info['name'], info['examples'])
#         elif info["chart_type"] == "wordcloud":
#             example_sentences = element_example_sentences.get(info["name"], [])

#             if info["name"] == "솔루션 제시":
#                 render_wordcloud(info["name"], solution_keywords, example_sentences)
#             else:
#                 keyword_freq = {kw: random.randint(10, 30) for kw in info["examples"]}
#                 render_wordcloud(info["name"], keyword_freq, example_sentences)
#         elif info["chart_type"] == "treemap":
#             render_treemap()
#         elif info["chart_type"] == "radar":
#             render_radar_chart()

# with st.container():
#     st.markdown("""
#     <div style='
#         border: 2px solid #e0e0e0;
#         border-radius: 10px;
#         padding: 20px;
#         background-color: #fafafa;
#         margin-bottom: 20px;
#     '>
#         <h3 style='margin-top: 0;'>대표적인 요소 구성 순서</h3>
#         <div style='font-size:18px; line-height:2.2'>
#             <b>문제 제기</b> →
#             <b>솔루션 제시</b> →
#             <b>제품 전달 가치</b> →
#             <b>제품 상세 설명</b> →
#             <b>브랜드 소개</b> →
#             <b>제품 및 브랜드 외부 평가</b> →
#             <b>펀딩 참여 유도</b> →
#             <b>자주 묻는 질문</b>
#         </div>
#     </div>
#     """, unsafe_allow_html=True)

# 🔻 요소별 분석 탭 레이아웃
element_tabs = st.tabs([info["name"] for info in element_analysis_info])

for i, info in enumerate(element_analysis_info):
    name = info["name"]
    chart_type = info["chart_type"]
    examples = info["examples"]

    with element_tabs[i]:
        #st.markdown(f"### 🔸 {name}")
        
        if chart_type == "pie":
            render_pie_chart(name, examples)

        elif chart_type == "wordcloud":
            example_sentences = element_example_sentences.get(name, [])

            if name == "솔루션 제시":
                render_wordcloud(name, solution_keywords, example_sentences)
            else:
                keyword_freq = {kw: random.randint(10, 30) for kw in examples}
                render_wordcloud(name, keyword_freq, example_sentences)

        elif chart_type == "treemap":
            render_treemap()

        elif chart_type == "radar":
            render_radar_chart()

# st.markdown("""
# <style>
# /* 전체 배경색 및 텍스트 색상 */
# body, .stApp {
#     background-color: #0D0C2B !important;
#     color: white !important;
# }

# /* 박스 스타일 */
# .custom-box {
#     border: 1px solid #444;
#     border-radius: 10px;
#     padding: 20px;
#     background-color: #111111;
#     margin-bottom: 20px;
#     color: white;
# }
# .custom-box h4 {
#     margin-top: 0;
#     font-size: 18px;
#     font-weight: 600;
#     color: #72F0FF;
# }

# /* 표(table) 스타일 */
# table {
#     border-collapse: collapse;
#     width: 100%;
# }
# th, td {
#     border: 1px solid #555;
#     padding: 6px 10px;
#     text-align: left;
#     font-size: 14px;
#     color: white;
# }
            

# /* ✅ 컬럼 헤더만 네온 하늘색 */
# th {
#     background-color: white;
#     color: #72F0FF;
# }

# }
# </style>
# """, unsafe_allow_html=True)

