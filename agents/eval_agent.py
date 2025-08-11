# agents/factor_agent.py

# from .base_agent import BaseAgent
# from clients.llm_client import LLMClient
# from clients.database_client import DatabaseClient
# from foundations.factor_structure import FactorParser, ComplexityAnalyzer, OriginalityAnalyzer

# class FactorAgent(BaseAgent):
#     """
#     주어진 가설을 바탕으로 알파 팩터를 생성하고 검증하는 에이전트입니다.
#     """
#     def __init__(self, llm_client: LLMClient, db_client: DatabaseClient):
#         self.llm_client = llm_client
#         self.db_client = db_client
        
#         # 팩터 분석을 위한 도구 초기화
#         self.parser = FactorParser()
#         self.complexity_analyzer = ComplexityAnalyzer()
#         self.originality_analyzer = OriginalityAnalyzer(self.parser, self.complexity_analyzer)
        
#         # 팩터 검증을 위한 임계값 설정
#         self.max_complexity_sl = 30  # 최대 상징적 길이
#         self.max_complexity_pc = 5   # 최대 파라미터 개수
#         self.max_similarity = 0.9    # 최대 유사도 (이 값 이상이면 너무 유사하여 탈락)
#         self.min_alignment = 0.6     # 최소 가설-설명-공식 일치도

#     def run(self):
#         """
#         데이터베이스에 있는 새로운 가설들을 팩터로 변환합니다.
#         """
#         print("\n--- FactorAgent 실행: 가설 기반 팩터 생성 시작 ---")
#         new_hypotheses = self.db_client.get_new_hypotheses()
        
#         if not new_hypotheses:
#             print("FactorAgent: 처리할 새로운 가설이 없습니다.")
#             print("--- FactorAgent 실행 종료 ---\n")
#             return

#         for hypothesis_record in new_hypotheses:
#             hyp_id = hypothesis_record['id']
#             hyp_data = hypothesis_record['data']
#             print(f"\n[가설 #{hyp_id} 처리 중]: {hyp_data['hypothesis']}")
#             self.db_client.update_hypothesis_status(hyp_id, 'processing')
            
#             # 1. 가설로부터 팩터 생성 (LLM)
#             factor_candidate = self.llm_client.generate_factor_from_hypothesis(hyp_data)
#             description = factor_candidate['description']
#             formula = factor_candidate['formula']
#             print(f"  - 생성된 공식: {formula}")

#             # 2. 팩터 파싱 및 분석
#             try:
#                 ast = self.parser.parse(formula)
#             except ValueError as e:
#                 print(f"  - ❌ 파싱 실패: {e}")
#                 continue

#             # 3. 정규화 지표 계산
#             sl = self.complexity_analyzer.calculate_symbolic_length(ast)
#             pc = self.complexity_analyzer.calculate_parameter_count(ast)
#             originality = self.originality_analyzer.calculate_similarity_score(ast)
            
#             align_h_d = self.llm_client.score_hypothesis_alignment(hyp_data['hypothesis'], description)
#             align_d_f = self.llm_client.score_description_alignment(description, formula)
#             # 두 정렬 점수의 기하평균으로 최종 점수 계산
#             alignment_score = (align_h_d['score'] * align_d_f['score']) ** 0.5
            
#             print(f"  - 복잡도(길이/파라미터): {sl}/{pc} | 유사도: {originality:.2f} | 일치도: {alignment_score:.2f}")

#             # 4. 팩터 유효성 검증
#             if sl > self.max_complexity_sl:
#                 print(f"  - ❌ 검증 실패: 복잡도(길이) 초과 ({sl} > {self.max_complexity_sl})")
#             elif pc > self.max_complexity_pc:
#                 print(f"  - ❌ 검증 실패: 복잡도(파라미터) 초과 ({pc} > {self.max_complexity_pc})")
#             elif originality > self.max_similarity:
#                 print(f"  - ❌ 검증 실패: 유사도 초과 ({originality:.2f} > {self.max_similarity})")
#             elif alignment_score < self.min_alignment:
#                 print(f"  - ❌ 검증 실패: 일치도 미달 ({alignment_score:.2f} < {self.min_alignment})")
#             else:
#                 # 5. 검증 통과 시 데이터베이스에 저장
#                 factor_data = {
#                     'hypothesis_id': hyp_id,
#                     'description': description,
#                     'formula': formula,
#                     'ast': ast,
#                     'complexity_sl': sl,
#                     'complexity_pc': pc,
#                     'originality_score': originality,
#                     'alignment_score': alignment_score,
#                 }
#                 factor_id = self.db_client.save_factor(factor_data)
#                 print(f"  - ✅ 검증 통과: 새로운 팩터 #{factor_id} 저장 완료.")

#             self.db_client.update_hypothesis_status(hyp_id, 'done')
        
#         print("\n--- FactorAgent 실행 종료 ---\n")


# agents/eval_agent.py

from .base_agent import BaseAgent
from clients.database_client import DatabaseClient
from clients.backtester_client import BacktesterClient

class EvalAgent(BaseAgent):
    """
    새롭게 생성된 팩터들의 성과를 백테스팅하여 평가하는 에이전트입니다.
    """
    def __init__(self, db_client: DatabaseClient, backtester_client: BacktesterClient):
        self.db_client = db_client
        self.backtester_client = backtester_client

    def run(self):
        print("\n--- EvalAgent 실행: 신규 팩터 평가 시작 ---")
        new_factors = self.db_client.get_new_factors()
        
        if not new_factors:
            print("EvalAgent: 평가할 새로운 팩터가 없습니다.")
            print("--- EvalAgent 실행 종료 ---\n")
            return

        for factor_record in new_factors:
            factor_id = factor_record['id']
            formula = factor_record['formula']
            ast = factor_record['ast']
            
            print(f"\n[팩터 #{factor_id} 평가 중]: {formula}")
            self.db_client.update_factor_status(factor_id, 'evaluating')

            try:
                # 💡 수정: backtester_client의 로직에 맞게 먼저 팩터 값을 계산합니다.
                factor_values = self.backtester_client.calculate_factor_values(formula, ast)
                performance_metrics = self.backtester_client.run_full_backtest(factor_values)
                
                eval_data = {'factor_id': factor_id, **performance_metrics}
                self.db_client.save_evaluation(eval_data)

                print(f"  - ✅ 평가 완료: IR {performance_metrics.get('IR'):.3f}, MDD {performance_metrics.get('MDD'):.3f}")

            except Exception as e:
                print(f"  - ❌ 평가 실패: {e}")
                self.db_client.update_factor_status(factor_id, 'failed')
        
        print("\n--- EvalAgent 실행 종료 ---\n")
