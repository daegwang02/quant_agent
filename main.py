
# import streamlit as st
# import pandas as pd
# import numpy as np

# # --- 1. 최종 코드의 모든 모듈 및 클래스 import ---
# from agents.idea_agent import IdeaAgent
# from agents.factor_agent import FactorAgent
# from agents.eval_agent import EvalAgent
# from agents.advisory_agent import AdvisoryAgent
# from clients.llm_client import LLMClient
# from clients.database_client import DatabaseClient
# from clients.backtester_client import BacktesterClient
# from optimizer.hyperparameter_optimizer import HyperparameterOptimizer # 추가
# # import config # config.py 파일은 Secrets로 대체되었으므로 주석 처리

# # --- 2. 페이지 디자인 및 구성 ---
# st.set_page_config(
#     page_title="Vibe Quant",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    
#     html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
#     .main-header h1 { color: #FFC107; text-align: center; font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
#     .stButton>button { background-color: #FFC107; color: black; border-radius: 8px; font-weight: 700; border: none; }
#     .stButton>button:hover { background-color: #E6B800; }
#     .stMetric > div { background-color: #f7f7f7; padding: 1.5rem; border-radius: 8px; border: 1px solid #ddd; }
#     .stMetric label { font-size: 1rem; color: #666; font-weight: normal; }
#     .stMetric p { font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; }
#     .report-header { color: #FFC107; font-weight: 700; border-bottom: 2px solid #FFC107; padding-bottom: 0.5rem; }
#     .stCodeBlock pre { background-color: #f0f0f0; border-left: 5px solid #FFC107; }
#     .streamlit-expander { border-left: 5px solid #FFC107; border-radius: 8px; }
#     </style>
#     <div class="main-header">
#         <h1>🤖 Vibe Quant</h1>
#     </div>
#     <br>
#     """, unsafe_allow_html=True)



# # --- 3. 세션 상태 초기화 ---
# if 'agents' not in st.session_state: st.session_state.agents = None
# if 'db' not in st.session_state: st.session_state.db = None
# if 'final_report' not in st.session_state: st.session_state.final_report = None
# if 'best_factor_info' not in st.session_state: st.session_state.best_factor_info = None
# if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

# # --- 4. 사이드바 (설정) ---
# with st.sidebar:
#     st.header("⚙️ 설정")
#     # 💡 API 키 설명을 OpenAI에 맞게 수정
#     st.info("API 키는 Secrets 설정에 `OPENAI_API_KEY`로 등록되어야 합니다.")

#     external_knowledge = st.text_area(
#         "💡 AI에게 제공할 시장 분석 정보 (선택)",
#         value="""최근 한국 주식 시장은 변동성이 크며, 특정 테마에 대한 쏠림 현상 이후 수급이 분산되고 있습니다. 거래량이 급증하며 특정 가격대를 돌파하는 종목들이 단기적으로 강한 시세를 보이는 경향이 있습니다.""",
#         height=150
#     )

#     discovery_rounds = st.number_input(
#         "🔄 알파 탐색 라운드 수",
#         min_value=1,
#         max_value=20,
#         value=5
#     )

#     run_optimization = st.checkbox("🧠 하이퍼파라미터 최적화 실행", value=False)

#     start_button = st.button("✨ 분석 시작!")
#     st.markdown("---")
#     st.info("데이터 파일 URL: Secrets의 `KOR_STOCK_DATA_URL`")

# # --- 5. 메인 화면 (분석 실행 및 결과 표시) ---
# st.write("### AI 투자 아이디어 입력")
# user_idea = st.text_area(
#     "어떤 투자 아이디어를 탐색하고 싶으신가요?",
#     "최근 3개월간 꾸준히 상승 추세를 보였으나, 단기적인 과열 신호(예: RSI 70 이상)가 없는 주식. 동시에 기업 가치 대비 저평가되어 있는 종목을 찾고 싶습니다",
#     height=80
# )

# # 분석 시작 버튼이 눌렸을 때 전체 워크플로우 실행
# if start_button:
#     # 세션 상태 초기화
#     st.session_state.analysis_done = True
#     st.session_state.final_report = None
#     st.session_state.best_factor_info = None

#     try:
#         # 💡 st.secrets에서 OPENAI_API_KEY를 불러와 LLMClient에 전달
#         llm_client = LLMClient(api_key=st.secrets.OPENAI_API_KEY)
        
#         # 💡 DatabaseClient와 BacktesterClient 초기화 코드는 그대로 유지
#         db_client = DatabaseClient(
#             data_url=st.secrets.KOR_STOCK_DATA_URL,
#             transaction_fee_buy=st.secrets.TRANSACTION_FEE_BUY,
#             transaction_fee_sell=st.secrets.TRANSACTION_FEE_SELL
#         )
        
#         backtester_client = BacktesterClient(
#             data_url=st.secrets.KOR_STOCK_DATA_URL,
#             transaction_fee_buy=st.secrets.TRANSACTION_FEE_BUY,
#             transaction_fee_sell=st.secrets.TRANSACTION_FEE_SELL
#         )
        
#         st.session_state.agents = {
#             'llm': llm_client,
#             'db': db_client,
#             'backtester': backtester_client,
#             'idea': IdeaAgent(llm_client, db_client),
#             'factor': FactorAgent(llm_client, db_client),
#             'eval': EvalAgent(db_client, backtester_client),
#             'advisory': AdvisoryAgent(llm_client, db_client)
#         }
#         st.session_state.db = db_client
#     except Exception as e:
#         st.error(f"초기화 오류: {e}. Secrets 설정이 올바른지 확인해주세요.")
#         st.stop()

#     # 2. 하이퍼파라미터 최적화 (선택 사항)
#     if run_optimization:
#         with st.spinner("🧠 하이퍼파라미터 최적화 중..."):
#             optimizer = HyperparameterOptimizer(
#                 st.session_state.agents['idea'],
#                 st.session_state.agents['factor'],
#                 st.session_state.agents['eval'],
#                 external_knowledge
#             )
#             best_params = optimizer.optimize(init_points=3, n_iter=5)
#             st.session_state.agents['factor'].max_complexity_sl = int(best_params['max_complexity_sl'])
#             st.session_state.agents['factor'].max_complexity_pc = int(best_params['max_complexity_pc'])
#             st.session_state.agents['factor'].max_similarity = best_params['max_similarity']
#             st.session_state.agents['factor'].min_alignment = best_params['min_alignment']
#             st.success("✅ 하이퍼파라미터 최적화 완료! 최적의 설정으로 분석을 시작합니다.")

#     # 3. 알파 탐색 루프 (실시간 상태 표시)
#     log_container = st.empty()
#     all_logs = []
    
#     with st.status("🚀 AlphaAgent 분석 시작...", expanded=True) as status:
#         current_knowledge = user_idea + "\n\n" + external_knowledge
        
#         for i in range(discovery_rounds):
#             log_container.info(f"🔄 **라운드 {i+1}/{discovery_rounds} 시작**")
#             status.update(label=f"🔄 라운드 {i+1}/{discovery_rounds} 진행 중...", state="running")
            
#             try:
#                 with st.spinner("💡 가설 생성 중..."):
#                     st.session_state.agents['idea'].run(current_knowledge)
#                     all_logs.append("💡 가설 생성 완료.")

#                 with st.spinner("📝 팩터 생성 및 검증 중..."):
#                     st.session_state.agents['factor'].run()
#                     all_logs.append("📝 팩터 생성 및 검증 완료.")

#                 with st.spinner("📊 팩터 백테스팅 및 평가 중..."):
#                     st.session_state.agents['eval'].run()
#                     all_logs.append("📊 팩터 백테스팅 완료.")
                
#                 log_container.success(f"✅ **라운드 {i+1}/{discovery_rounds} 성공!**")
                
#             except Exception as e:
#                 log_container.error(f"❌ **라운드 {i+1} 실패!** 오류: {e}")
#                 status.update(label=f"❌ 오류 발생! 라운드 {i+1} 중단", state="error")
#                 st.session_state.final_report = f"분석 중 오류 발생: {e}"
#                 st.session_state.best_factor_info = None
#                 break
        
#         if st.session_state.final_report is None:
#             status.update(label="📜 최종 투자 조언 리포트 생성 중...", state="running")
#             best_factor_info = st.session_state.db.get_best_factor()
#             if best_factor_info:
#                 st.session_state.best_factor_info = best_factor_info
#                 llm_client = st.session_state.agents['llm']
#                 st.session_state.final_report = llm_client.generate_investment_advice(best_factor_info)
#                 status.update(label="🎉 분석 완료! 최종 리포트가 생성되었습니다.", state="complete", expanded=False)
#             else:
#                 st.error("분석을 통해 유의미한 팩터를 찾지 못했습니다.")
#                 status.update(label="분석 실패.", state="error")

# main.py

# import streamlit as st
# import pandas as pd
# import numpy as np
# import warnings

# warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# # --- 1. 최종 코드의 모든 모듈 및 클래스 import ---
# from agents.idea_agent import IdeaAgent
# from agents.factor_agent import FactorAgent
# from agents.eval_agent import EvalAgent
# from agents.advisory_agent import AdvisoryAgent
# from clients.llm_client import LLMClient
# from clients.database_client import DatabaseClient
# from clients.backtester_client import BacktesterClient
# from optimizer.hyperparameter_optimizer import HyperparameterOptimizer
# from foundations.factor_structure import FactorParser, ComplexityAnalyzer, OriginalityAnalyzer

# # --- 2. 페이지 디자인 및 구성 ---
# st.set_page_config(
#     page_title="Vibe Quant",
#     page_icon="📈",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# st.markdown("""
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    
#     html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
#     .main-header h1 { color: #FFC107; text-align: center; font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
#     .stButton>button { background-color: #FFC107; color: black; border-radius: 8px; font-weight: 700; border: none; }
#     .stButton>button:hover { background-color: #E6B800; }
#     .stMetric > div { background-color: #f7f7f7; padding: 1.5rem; border-radius: 8px; border: 1px solid #ddd; }
#     .stMetric label { font-size: 1rem; color: #666; font-weight: normal; }
#     .stMetric p { font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; }
#     .report-header { color: #FFC107; font-weight: 700; border-bottom: 2px solid #FFC107; padding-bottom: 0.5rem; }
#     .stCodeBlock pre { background-color: #f0f0f0; border-left: 5px solid #FFC107; }
#     .streamlit-expander { border-left: 5px solid #FFC107; border-radius: 8px; }
#     </style>
#     <div class="main-header">
#         <h1>🤖 Vibe Quant</h1>
#     </div>
#     <br>
#     """, unsafe_allow_html=True)


# # --- 3. 세션 상태 초기화 ---
# if 'agents' not in st.session_state: st.session_state.agents = None
# if 'db' not in st.session_state: st.session_state.db = None
# if 'final_report' not in st.session_state: st.session_state.final_report = None
# if 'best_factor_info' not in st.session_state: st.session_state.best_factor_info = None
# if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

# # --- 4. 사이드바 (설정) ---
# with st.sidebar:
#     st.header("⚙️ 설정")
#     st.info("API 키는 Secrets 설정에 `OPENAI_API_KEY`로 등록되어야 합니다.")

#     external_knowledge = st.text_area(
#         "💡 AI에게 제공할 시장 분석 정보 (선택)",
#         value="""최근 한국 주식 시장은 변동성이 크며, 특정 테마에 대한 쏠림 현상 이후 수급이 분산되고 있습니다. 거래량이 급증하며 특정 가격대를 돌파하는 종목들이 단기적으로 강한 시세를 보이는 경향이 있습니다.""",
#         height=150
#     )

#     discovery_rounds = st.number_input(
#         "🔄 알파 탐색 라운드 수",
#         min_value=1,
#         max_value=20,
#         value=5
#     )

#     run_optimization = st.checkbox("🧠 하이퍼파라미터 최적화 실행", value=False)
    
#     if not run_optimization:
#         st.subheader("팩터 생성 조건 (수동 설정)")
#         st.slider("상징적 길이 (max_complexity_sl)", min_value=10, max_value=50, value=25, key="manual_sl")
#         st.slider("파라미터 개수 (max_complexity_pc)", min_value=2, max_value=10, value=5, key="manual_pc")
#         st.slider("최대 유사도 (max_similarity)", min_value=0.5, max_value=1.0, value=0.9, step=0.01, key="manual_sim")
#         st.slider("최소 일치도 (min_alignment)", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="manual_align")

#     start_button = st.button("✨ 분석 시작!")
#     st.markdown("---")
#     st.info("데이터 파일 URL: Secrets의 `KOR_STOCK_DATA_URL`")

# # --- 5. 메인 화면 (분석 실행 및 결과 표시) ---
# st.write("### AI 투자 아이디어 입력")
# user_idea = st.text_area(
#     "어떤 투자 아이디어를 탐색하고 싶으신가요?",
#     "최근 3개월간 꾸준히 상승 추세를 보였으나, 단기적인 과열 신호(예: RSI 70 이상)가 없는 주식. 동시에 기업 가치 대비 저평가되어 있는 종목을 찾고 싶습니다",
#     height=80
# )

# if start_button:
#     st.session_state.analysis_done = True
#     st.session_state.final_report = None
#     st.session_state.best_factor_info = None

#     try:
#         llm_client = LLMClient(api_key=st.secrets.OPENAI_API_KEY)
        
#         db_client = DatabaseClient()
        
#         backtester_client = BacktesterClient(
#             data_url=st.secrets.KOR_STOCK_DATA_URL,
#             transaction_fee_buy=st.secrets.TRANSACTION_FEE_BUY,
#             transaction_fee_sell=st.secrets.TRANSACTION_FEE_SELL
#         )
        
#         parser = FactorParser()
#         complexity_analyzer = ComplexityAnalyzer()
#         originality_analyzer = OriginalityAnalyzer(parser, complexity_analyzer)
        
#         st.session_state.agents = {
#             'llm': llm_client,
#             'db': db_client,
#             'backtester': backtester_client,
#             'idea': IdeaAgent(llm_client, db_client),
#             'factor': FactorAgent(llm_client, db_client, parser, complexity_analyzer, originality_analyzer),
#             'eval': EvalAgent(db_client, backtester_client),
#             'advisory': AdvisoryAgent(llm_client, db_client)
#         }
#         st.session_state.db = db_client

#     except Exception as e:
#         st.error(f"초기화 오류: {e}. Secrets 설정이 올바른지 확인해주세요.")
#         st.stop()

#     if run_optimization:
#         with st.spinner("🧠 하이퍼파라미터 최적화 중..."):
#             optimizer = HyperparameterOptimizer(
#                 st.session_state.agents['idea'],
#                 st.session_state.agents['factor'],
#                 st.session_state.agents['eval'],
#                 external_knowledge
#             )
#             best_params = optimizer.optimize(init_points=3, n_iter=5)
#             st.session_state.agents['factor'].max_complexity_sl = int(best_params['max_complexity_sl'])
#             st.session_state.agents['factor'].max_complexity_pc = int(best_params['max_complexity_pc'])
#             st.session_state.agents['factor'].max_similarity = best_params['max_similarity']
#             st.session_state.agents['factor'].min_alignment = best_params['min_alignment']
#             st.success("✅ 하이퍼파라미터 최적화 완료! 최적의 설정으로 분석을 시작합니다.")
#     else:
#         st.session_state.agents['factor'].max_complexity_sl = st.session_state.manual_sl
#         st.session_state.agents['factor'].max_complexity_pc = st.session_state.manual_pc
#         st.session_state.agents['factor'].max_similarity = st.session_state.manual_sim
#         st.session_state.agents['factor'].min_alignment = st.session_state.manual_align
#         st.success("✅ 수동 설정된 값으로 분석을 시작합니다.")

#     log_container = st.empty()
#     all_logs = []
    
#     with st.status("🚀 AlphaAgent 분석 시작...", expanded=True) as status:
#         current_knowledge = user_idea + "\n\n" + external_knowledge
        
#         for i in range(discovery_rounds):
#             log_container.info(f"🔄 **라운드 {i+1}/{discovery_rounds} 시작**")
#             status.update(label=f"🔄 라운드 {i+1}/{discovery_rounds} 진행 중...", state="running")
            
#             try:
#                 feedback_summary = st.session_state.db.get_evaluation_summary()

#                 with st.spinner("💡 가설 생성 중..."):
#                     st.session_state.agents['idea'].run(current_knowledge, feedback_summary)
#                     all_logs.append("💡 가설 생성 완료.")

#                 with st.spinner("📝 팩터 생성 및 검증 중..."):
#                     st.session_state.agents['factor'].run()
#                     all_logs.append("📝 팩터 생성 및 검증 완료.")

#                 with st.spinner("📊 팩터 백테스팅 및 평가 중..."):
#                     st.session_state.agents['eval'].run()
#                     all_logs.append("📊 팩터 백테스팅 완료.")
                
#                 log_container.success(f"✅ **라운드 {i+1}/{discovery_rounds} 성공!**")
                
#             except Exception as e:
#                 log_container.error(f"❌ **라운드 {i+1} 실패!** 오류: {e}")
#                 status.update(label=f"❌ 오류 발생! 라운드 {i+1} 중단", state="error")
#                 st.session_state.final_report = f"분석 중 오류 발생: {e}"
#                 st.session_state.best_factor_info = None
#                 break
        
#         if st.session_state.final_report is None:
#             status.update(label="📜 최종 투자 조언 리포트 생성 중...", state="running")
#             best_factor_info = st.session_state.db.get_best_factor()
#             if best_factor_info:
#                 st.session_state.best_factor_info = best_factor_info
#                 llm_client = st.session_state.agents['llm']
#                 st.session_state.final_report = llm_client.generate_investment_advice(best_factor_info)
#                 status.update(label="🎉 분석 완료! 최종 리포트가 생성되었습니다.", state="complete", expanded=False)
#             else:
#                 st.error("분석을 통해 유의미한 팩터를 찾지 못했습니다.")
#                 status.update(label="분석 실패.", state="error")

import streamlit as st
import pandas as pd
import numpy as np
import warnings
import httpx  # httpx 라이브러리 추가

warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)


# --- 1. 최종 코드의 모든 모듈 및 클래스 import ---
from agents.idea_agent import IdeaAgent
from agents.factor_agent import FactorAgent
from agents.eval_agent import EvalAgent
from agents.advisory_agent import AdvisoryAgent
from clients.llm_client import LLMClient
from clients.database_client import DatabaseClient
from clients.backtester_client import BacktesterClient
from optimizer.hyperparameter_optimizer import HyperparameterOptimizer
from foundations.factor_structure import FactorParser, ComplexityAnalyzer, OriginalityAnalyzer

# --- 2. 페이지 디자인 및 구성 ---
st.set_page_config(
    page_title="Vibe Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@400;700&display=swap');
    
    html, body, [class*="st-"] { font-family: 'Noto Sans KR', sans-serif; }
    .main-header h1 { color: #FFC107; text-align: center; font-size: 2.5rem; font-weight: 700; margin-bottom: 0; }
    .stButton>button { background-color: #FFC107; color: black; border-radius: 8px; font-weight: 700; border: none; }
    .stButton>button:hover { background-color: #E6B800; }
    .stMetric > div { background-color: #f7f7f7; padding: 1.5rem; border-radius: 8px; border: 1px solid #ddd; }
    .stMetric label { font-size: 1rem; color: #666; font-weight: normal; }
    .stMetric p { font-size: 1.5rem; font-weight: 700; margin-top: 0.5rem; }
    .report-header { color: #FFC107; font-weight: 700; border-bottom: 2px solid #FFC107; padding-bottom: 0.5rem; }
    .stCodeBlock pre { background-color: #f0f0f0; border-left: 5px solid #FFC107; }
    .streamlit-expander { border-left: 5px solid #FFC107; border-radius: 8px; }
    </style>
    <div class="main-header">
        <h1>🤖 Vibe Quant</h1>
    </div>
    <br>
    """, unsafe_allow_html=True)


# --- 3. 세션 상태 초기화 ---
if 'agents' not in st.session_state: st.session_state.agents = None
if 'db' not in st.session_state: st.session_state.db = None
if 'final_report' not in st.session_state: st.session_state.final_report = None
if 'best_factor_info' not in st.session_state: st.session_state.best_factor_info = None
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False

# --- 4. 사이드바 (설정) ---
with st.sidebar:
    st.header("⚙️ 설정")
    st.info("API 키는 Secrets 설정에 `OPENAI_API_KEY`로 등록되어야 합니다.")

    external_knowledge = st.text_area(
        "💡 AI에게 제공할 시장 분석 정보 (선택)",
        value="""최근 한국 주식 시장은 변동성이 크며, 특정 테마에 대한 쏠림 현상 이후 수급이 분산되고 있습니다. 거래량이 급증하며 특정 가격대를 돌파하는 종목들이 단기적으로 강한 시세를 보이는 경향이 있습니다.""",
        height=150
    )

    discovery_rounds = st.number_input(
        "🔄 알파 탐색 라운드 수",
        min_value=1,
        max_value=20,
        value=5
    )

    run_optimization = st.checkbox("🧠 하이퍼파라미터 최적화 실행", value=False)
    
    if not run_optimization:
        st.subheader("팩터 생성 조건 (수동 설정)")
        st.slider("상징적 길이 (max_complexity_sl)", min_value=10, max_value=50, value=25, key="manual_sl")
        st.slider("파라미터 개수 (max_complexity_pc)", min_value=2, max_value=10, value=5, key="manual_pc")
        st.slider("최대 유사도 (max_similarity)", min_value=0.5, max_value=1.0, value=0.9, step=0.01, key="manual_sim")
        st.slider("최소 일치도 (min_alignment)", min_value=0.0, max_value=1.0, value=0.7, step=0.01, key="manual_align")

    start_button = st.button("✨ 분석 시작!")
    st.markdown("---")
    st.info("데이터 파일 URL: Secrets의 `KOR_STOCK_DATA_URL`")

# --- 5. 메인 화면 (분석 실행 및 결과 표시) ---
st.write("### AI 투자 아이디어 입력")
user_idea = st.text_area(
    "어떤 투자 아이디어를 탐색하고 싶으신가요?",
    "최근 3개월간 꾸준히 상승 추세를 보였으나, 단기적인 과열 신호(예: RSI 70 이상)가 없는 주식. 동시에 기업 가치 대비 저평가되어 있는 종목을 찾고 싶습니다",
    height=80
)

if start_button:
    st.session_state.analysis_done = True
    st.session_state.final_report = None
    st.session_state.best_factor_info = None

    try:
        # secrets에서 proxies 설정을 확인하고 LLMClient에 전달합니다.
        proxies = st.secrets.get("proxies", None)
        llm_client = LLMClient(api_key=st.secrets.OPENAI_API_KEY, proxies=proxies)
        
        db_client = DatabaseClient()
        
        backtester_client = BacktesterClient(
            data_url=st.secrets.KOR_STOCK_DATA_URL,
            transaction_fee_buy=st.secrets.TRANSACTION_FEE_BUY,
            transaction_fee_sell=st.secrets.TRANSACTION_FEE_SELL
        )
        
        parser = FactorParser()
        complexity_analyzer = ComplexityAnalyzer()
        originality_analyzer = OriginalityAnalyzer(parser, complexity_analyzer)
        
        st.session_state.agents = {
            'llm': llm_client,
            'db': db_client,
            'backtester': backtester_client,
            'idea': IdeaAgent(llm_client, db_client),
            'factor': FactorAgent(llm_client, db_client, parser, complexity_analyzer, originality_analyzer),
            'eval': EvalAgent(db_client, backtester_client),
            'advisory': AdvisoryAgent(llm_client, db_client)
        }
        st.session_state.db = db_client

    except Exception as e:
        st.error(f"초기화 오류: {e}. Secrets 설정이 올바른지 확인해주세요.")
        st.stop()

    if run_optimization:
        with st.spinner("🧠 하이퍼파라미터 최적화 중..."):
            optimizer = HyperparameterOptimizer(
                st.session_state.agents['idea'],
                st.session_state.agents['factor'],
                st.session_state.agents['eval'],
                external_knowledge
            )
            best_params = optimizer.optimize(init_points=3, n_iter=5)
            st.session_state.agents['factor'].max_complexity_sl = int(best_params['max_complexity_sl'])
            st.session_state.agents['factor'].max_complexity_pc = int(best_params['max_complexity_pc'])
            st.session_state.agents['factor'].max_similarity = best_params['max_similarity']
            st.session_state.agents['factor'].min_alignment = best_params['min_alignment']
            st.success("✅ 하이퍼파라미터 최적화 완료! 최적의 설정으로 분석을 시작합니다.")
    else:
        st.session_state.agents['factor'].max_complexity_sl = st.session_state.manual_sl
        st.session_state.agents['factor'].max_complexity_pc = st.session_state.manual_pc
        st.session_state.agents['factor'].max_similarity = st.session_state.manual_sim
        st.session_state.agents['factor'].min_alignment = st.session_state.manual_align
        st.success("✅ 수동 설정된 값으로 분석을 시작합니다.")

    log_container = st.empty()
    all_logs = []
    
    with st.status("🚀 AlphaAgent 분석 시작...", expanded=True) as status:
        current_knowledge = user_idea + "\n\n" + external_knowledge
        
        for i in range(discovery_rounds):
            log_container.info(f"🔄 **라운드 {i+1}/{discovery_rounds} 시작**")
            status.update(label=f"🔄 라운드 {i+1}/{discovery_rounds} 진행 중...", state="running")
            
            try:
                feedback_summary = st.session_state.db.get_evaluation_summary()

                with st.spinner("💡 가설 생성 중..."):
                    st.session_state.agents['idea'].run(current_knowledge, feedback_summary)
                    all_logs.append("💡 가설 생성 완료.")

                with st.spinner("📝 팩터 생성 및 검증 중..."):
                    st.session_state.agents['factor'].run()
                    all_logs.append("📝 팩터 생성 및 검증 완료.")

                with st.spinner("📊 팩터 백테스팅 및 평가 중..."):
                    st.session_state.agents['eval'].run()
                    all_logs.append("📊 팩터 백테스팅 완료.")
                
                log_container.success(f"✅ **라운드 {i+1}/{discovery_rounds} 성공!**")
                
            except Exception as e:
                log_container.error(f"❌ **라운드 {i+1} 실패!** 오류: {e}")
                status.update(label=f"❌ 오류 발생! 라운드 {i+1} 중단", state="error")
                st.session_state.final_report = f"분석 중 오류 발생: {e}"
                st.session_state.best_factor_info = None
                break
        
        if st.session_state.final_report is None:
            status.update(label="📜 최종 투자 조언 리포트 생성 중...", state="running")
            best_factor_info = st.session_state.db.get_best_factor()
            if best_factor_info:
                st.session_state.best_factor_info = best_factor_info
                llm_client = st.session_state.agents['llm']
                st.session_state.final_report = llm_client.generate_investment_advice(best_factor_info)
                status.update(label="🎉 분석 완료! 최종 리포트가 생성되었습니다.", state="complete", expanded=False)
            else:
                st.error("분석을 통해 유의미한 팩터를 찾지 못했습니다.")
                status.update(label="분석 실패.", state="error")
