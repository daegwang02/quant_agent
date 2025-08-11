# # foundations/factor_structure.py

# import re
# from typing import List, Union, Any, Set
# # alpha_zoo.py에서 팩터 리스트를 가져오기 위해 import 추가
# from .alpha_zoo import get_alpha_zoo

# # --- AST 노드 클래스 정의 (이전 코드와 동일) ---
# class ASTNode:
#     """모든 AST 노드의 기본이 되는 추상 클래스입니다."""
#     def __repr__(self):
#         return self.__str__()
    
#     # hashable하게 만들어 set에 저장할 수 있도록 함
#     def __hash__(self):
#         return hash(str(self))

#     def __eq__(self, other):
#         return str(self) == str(other)

# class OperatorNode(ASTNode):
#     """연산자(e.g., rank, +, -)를 나타내는 노드입니다."""
#     def __init__(self, op: str, children: List[ASTNode]):
#         self.op = op
#         self.children = children

#     def __str__(self):
#         children_str = ', '.join(map(str, self.children))
#         return f"{self.op}({children_str})"

# class VariableNode(ASTNode):
#     """변수(e.g., close, volume)를 나타내는 노드입니다."""
#     def __init__(self, name: str):
#         self.name = name

#     def __str__(self):
#         return f"${self.name}"

# class LiteralNode(ASTNode):
#     """리터럴(상수 값, e.g., 10, 'sector')을 나타내는 노드입니다."""
#     def __init__(self, value: Any):
#         self.value = value

#     def __str__(self):
#         if isinstance(self.value, str):
#             return f'"{self.value}"'
#         return str(self.value)

# # --- 팩터 공식 파서 (이전 코드와 동일) ---
# class FactorParser:
#     """
#     팩터 공식 문자열을 AST로 변환하는 파서입니다.
#     """
#     # (이전 단계에서 작성한 FactorParser 클래스 코드가 여기에 위치합니다. 생략)
#     def __init__(self):
#         self.tokens = []
#         self.pos = 0

#     def parse(self, formula: str) -> ASTNode:
#         self.tokens = self._tokenize(formula)
#         self.pos = 0
#         ast = self._parse_expression()
#         if self.pos < len(self.tokens):
#             raise ValueError("파싱 후 남은 토큰이 있습니다: {}".format(self.tokens[self.pos:]))
#         return ast

#     def _tokenize(self, formula: str) -> List[str]:
#         token_regex = re.compile(r'([A-Za-z_][A-Za-z0-9_\.]*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[()+\-*/?^:,<>])')
#         tokens = token_regex.findall(formula)
#         return [token for token in tokens if token.strip() and token != '$']

#     def _peek(self) -> Union[str, None]:
#         return self.tokens[self.pos] if self.pos < len(self.tokens) else None

#     def _consume(self) -> str:
#         token = self._peek()
#         self.pos += 1
#         return token

#     def _parse_primary(self) -> ASTNode:
#         token = self._peek()
#         if token == '-':
#             self._consume()
#             return OperatorNode('neg', [self._parse_primary()])
#         if token.replace('.', '', 1).isdigit():
#             return LiteralNode(float(self._consume()))
#         if token.isalnum() or '_' in token or '.' in token:
#             self._consume()
#             if self._peek() == '(':
#                 self._consume()
#                 args = []
#                 if self._peek() != ')':
#                     while True:
#                         args.append(self._parse_expression())
#                         if self._peek() == ')':
#                             break
#                         if self._peek() != ',':
#                             raise ValueError("함수 인자 사이에 콤마(,)가 필요합니다.")
#                         self._consume()
#                 self._consume()
#                 return OperatorNode(token, args)
#             else:
#                 return VariableNode(token)
#         if token == '(':
#             self._consume()
#             expr = self._parse_expression()
#             if self._consume() != ')':
#                 raise ValueError("괄호가 닫히지 않았습니다.")
#             return expr
#         raise ValueError(f"예상치 못한 토큰입니다: {token}")

#     def _parse_binary_op(self, parse_next_level, ops: List[str]) -> ASTNode:
#         node = parse_next_level()
#         while self._peek() in ops:
#             op = self._consume()
#             right = parse_next_level()
#             node = OperatorNode(op, [node, right])
#         return node

#     def _parse_power(self) -> ASTNode:
#         return self._parse_binary_op(self._parse_primary, ['^'])

#     def _parse_multiplicative(self) -> ASTNode:
#         return self._parse_binary_op(self._parse_power, ['*', '/'])

#     def _parse_additive(self) -> ASTNode:
#         return self._parse_binary_op(self._parse_multiplicative, ['+', '-'])
        
#     def _parse_comparison(self) -> ASTNode:
#         return self._parse_binary_op(self._parse_additive, ['>', '<', '>=', '<=', '==', '!='])

#     def _parse_logical(self) -> ASTNode:
#         return self._parse_binary_op(self._parse_comparison, ['&&', '||'])

#     def _parse_expression(self) -> ASTNode:
#         node = self._parse_logical()
#         if self._peek() == '?':
#             self._consume()
#             true_expr = self._parse_expression()
#             if self._consume() != ':':
#                 raise ValueError("삼항 연산자에 ':'가 필요합니다.")
#             false_expr = self._parse_expression()
#             return OperatorNode('If', [node, true_expr, false_expr])
#         return node

# # --- 팩터 복잡도 분석기 (이전 코드와 동일) ---
# class ComplexityAnalyzer:
#     """AST를 기반으로 팩터의 복잡도를 계산합니다."""
#     # (이전 단계에서 작성한 ComplexityAnalyzer 클래스 코드가 여기에 위치합니다. 생략)
#     def calculate_symbolic_length(self, node: ASTNode) -> int:
#         if isinstance(node, OperatorNode):
#             return 1 + sum(self.calculate_symbolic_length(child) for child in node.children)
#         return 1

#     def calculate_parameter_count(self, node: ASTNode) -> int:
#         count = 0
#         if isinstance(node, LiteralNode) and isinstance(node.value, (int, float)):
#             count = 1
#         if isinstance(node, OperatorNode):
#             count += sum(self.calculate_parameter_count(child) for child in node.children)
#         return count


# # --- 팩터 독창성 분석기 (신규 추가) ---

# class OriginalityAnalyzer:
#     """
#     AST 비교를 통해 팩터의 독창성을 평가합니다.
#     Alpha Zoo에 있는 기존 팩터들과의 유사도를 측정합니다.
#     """
#     def __init__(self, parser: FactorParser, complexity_analyzer: ComplexityAnalyzer):
#         """
#         초기화 시 Alpha Zoo의 모든 팩터를 파싱하여 AST 형태로 저장해 둡니다.
#         """
#         self.parser = parser
#         self.complexity_analyzer = complexity_analyzer
#         self.alpha_zoo_asts = self._load_alpha_zoo_asts()

#     def _load_alpha_zoo_asts(self) -> List[ASTNode]:
#         """Alpha Zoo의 팩터 공식들을 AST로 변환하여 리스트로 반환합니다."""
#         formulas = get_alpha_zoo()
#         asts = []
#         for formula in formulas:
#             try:
#                 asts.append(self.parser.parse(formula))
#             except Exception as e:
#                 # 101 Alpha의 일부 표현식은 구현된 파서가 처리 못할 수 있으므로, 오류 발생 시 경고만 출력
#                 # print(f"경고: Alpha Zoo 팩터 '{formula}' 파싱 실패: {e}")
#                 pass
#         return asts

#     def _are_isomorphic(self, node1: ASTNode, node2: ASTNode) -> bool:
#         """두 AST (또는 서브트리)가 구조적으로 동일한지 재귀적으로 확인합니다."""
#         if type(node1) is not type(node2):
#             return False

#         if isinstance(node1, OperatorNode):
#             # OperatorNode인 경우
#             return (node1.op == node2.op and
#                     len(node1.children) == len(node2.children) and
#                     all(self._are_isomorphic(c1, c2) for c1, c2 in zip(node1.children, node2.children)))
#         elif isinstance(node1, VariableNode):
#             # VariableNode인 경우
#             return node1.name == node2.name
#         elif isinstance(node1, LiteralNode):
#             # LiteralNode인 경우
#             return node1.value == node2.value
#         return False

#     def _get_all_subtrees(self, node: ASTNode) -> Set[ASTNode]:
#         """주어진 AST에서 모든 가능한 서브트리들을 set 형태로 반환합니다."""
#         subtrees = {node}
#         if isinstance(node, OperatorNode):
#             for child in node.children:
#                 subtrees.update(self._get_all_subtrees(child))
#         return subtrees

#     def calculate_similarity_score(self, new_factor_ast: ASTNode) -> float:
#         """
#         새로운 팩터가 Alpha Zoo의 팩터들과 얼마나 유사한지 계산합니다.
#         가장 높은 유사도 점수를 반환합니다. (0: 완전 다름, 1: 완전 동일)

#         Args:
#             new_factor_ast (ASTNode): 독창성을 평가할 새로운 팩터의 AST입니다.

#         Returns:
#             float: Alpha Zoo 내 팩터들과의 최대 유사도 점수 (S(f)).
#         """
#         max_similarity = 0.0
#         new_factor_subtrees = self._get_all_subtrees(new_factor_ast)
#         size_new = self.complexity_analyzer.calculate_symbolic_length(new_factor_ast)

#         for zoo_ast in self.alpha_zoo_asts:
#             zoo_subtrees = self._get_all_subtrees(zoo_ast)
            
#             # 공통 서브트리 찾기
#             common_subtrees = new_factor_subtrees.intersection(zoo_subtrees)
            
#             if not common_subtrees:
#                 continue
            
#             # 가장 큰 공통 서브트리의 크기 계산
#             largest_common_subtree_size = 0
#             for sub in common_subtrees:
#                 # is_isomorphic은 set에 넣을 때 이미 암시적으로 검사되었지만, 명확성을 위해 로직 유지
#                 # 실제로는 common_subtrees에 있는 것만으로도 동일 구조임이 보장됨.
#                 size = self.complexity_analyzer.calculate_symbolic_length(sub)
#                 if size > largest_common_subtree_size:
#                     largest_common_subtree_size = size

#             # 정규화된 유사도 계산 (논문 Eq.5 참고, 단 분모는 max(|T(fi)|, |T(fj)|) 대신 두 사이즈의 평균 사용)
#             # 이는 한쪽이 매우 클 때 유사도가 과소평가되는 것을 방지하기 위함. 논문의 수식을 엄격히 따를 수도 있음.
#             size_zoo = self.complexity_analyzer.calculate_symbolic_length(zoo_ast)
#             # 논문의 수식을 더 정확히 따르기 위해 max로 분모를 계산
#             # similarity = largest_common_subtree_size / max(size_new, size_zoo)
            
#             # 좀 더 직관적인 정규화를 위해 두 트리의 크기 합으로 나눔
#             # 이는 두 트리의 공통 부분의 비율을 나타냄
#             similarity = (2 * largest_common_subtree_size) / (size_new + size_zoo)


#             if similarity > max_similarity:
#                 max_similarity = similarity
        
#         return max_similarity

# '''
# # --- 테스트 코드 ---
# if __name__ == '__main__':
#     parser = FactorParser()
#     complexity_analyzer = ComplexityAnalyzer()
    
#     # 1. 이전 테스트 실행
#     sample_formula = "(rank((open - (sum(vwap, 10) / 10))) * (-1 * abs(rank((close - vwap)))))"
#     print("="*50)
#     print(f"테스트 공식 1: {sample_formula}")
#     print("="*50)
#     ast_tree = parser.parse(sample_formula)
#     print("\n[파서 & 복잡도 테스트]")
#     print(f"AST: {ast_tree}")
#     print(f"상징적 길이: {complexity_analyzer.calculate_symbolic_length(ast_tree)}")
#     print(f"파라미터 개수: {complexity_analyzer.calculate_parameter_count(ast_tree)}")

#     # 2. 독창성 분석기 테스트
#     print("\n" + "="*50)
#     print("[독창성 분석기 테스트]")
#     print("="*50)
#     originality_analyzer = OriginalityAnalyzer(parser, complexity_analyzer)
#     print(f"Alpha Zoo 로드 완료: 총 {len(originality_analyzer.alpha_zoo_asts)}개 팩터 AST")
    
#     # 테스트 케이스 1: Alpha Zoo에 있는 팩터와 거의 동일한 팩터 (Alpha#6)
#     similar_formula = "(-1 * correlation(open, volume, 10))" # Alpha#6과 동일
#     similar_ast = parser.parse(similar_formula)
#     similarity1 = originality_analyzer.calculate_similarity_score(similar_ast)
#     print(f"\n테스트 공식 2 (유사도 높음 예상): {similar_formula}")
#     print(f"-> Alpha Zoo와의 유사도 점수: {similarity1:.4f} (1.0에 가까울수록 유사)")

#     # 테스트 케이스 2: Alpha Zoo에 있는 팩터와 구조가 약간 다른 팩터
#     modified_formula = "(-1 * correlation(high, volume, 10))" # Alpha#6에서 open을 high로 변경
#     modified_ast = parser.parse(modified_formula)
#     similarity2 = originality_analyzer.calculate_similarity_score(modified_ast)
#     print(f"\n테스트 공식 3 (유사도 중간 예상): {modified_formula}")
#     print(f"-> Alpha Zoo와의 유사도 점수: {similarity2:.4f}")

#     # 테스트 케이스 3: 완전히 새로운 구조의 팩터
#     new_formula = "rank(ts_corr(rank(low), rank(adv20), 5))" # ts_corr은 가정. correlation으로 테스트
#     new_formula_for_test = "rank(correlation(rank(low), rank(adv20), 5))" 
#     new_ast = parser.parse(new_formula_for_test)
#     similarity3 = originality_analyzer.calculate_similarity_score(new_ast)
#     print(f"\n테스트 공식 4 (유사도 낮음 예상): {new_formula_for_test}")
#     print(f"-> Alpha Zoo와의 유사도 점수: {similarity3:.4f} (0에 가까울수록 독창적)")
# '''

# foundations/factor_structure.py
# foundations/factor_structure.py

import re
from typing import List, Union, Any, Set
from .alpha_zoo import get_alpha_zoo

# --- AST 노드 클래스 정의 (이전 코드와 동일) ---
class ASTNode:
    def __repr__(self):
        return self.__str__()
    
    def __hash__(self):
        return hash(str(self))

    def __eq__(self, other):
        return str(self) == str(other)

class OperatorNode(ASTNode):
    def __init__(self, op: str, children: List[ASTNode]):
        self.op = op
        self.children = children

    def __str__(self):
        children_str = ', '.join(map(str, self.children))
        return f"{self.op}({children_str})"

class VariableNode(ASTNode):
    def __init__(self, name: str):
        self.name = name

    def __str__(self):
        return f"${self.name}"

class LiteralNode(ASTNode):
    def __init__(self, value: Any):
        self.value = value

    def __str__(self):
        if isinstance(self.value, str):
            return f'"{self.value}"'
        return str(self.value)

# --- 팩터 공식 파서 ---
class FactorParser:
    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, formula: str) -> ASTNode:
        self.tokens = self._tokenize(formula)
        self.pos = 0
        ast = self._parse_expression()
        if self.pos < len(self.tokens):
            raise ValueError("파싱 후 남은 토큰이 있습니다: {}".format(self.tokens[self.pos:]))
        return ast

    def _tokenize(self, formula: str) -> List[str]:
        # 💡 삼항 연산자를 포함한 모든 연산자를 정확히 파싱하도록 정규식을 수정합니다.
        token_regex = re.compile(r'([A-Za-z_][A-Za-z0-9_\.]*|\d+\.?\d*|==|!=|<=|>=|&&|\|\||[()+\-*/?^:,<>])')
        tokens = token_regex.findall(formula)
        return [token for token in tokens if token.strip() and token != '$']

    def _peek(self) -> Union[str, None]:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> str:
        token = self._peek()
        self.pos += 1
        return token

    def _parse_primary(self) -> ASTNode:
        token = self._peek()
        if token == '-':
            self._consume()
            return OperatorNode('neg', [self._parse_primary()])
        if token.replace('.', '', 1).isdigit():
            return LiteralNode(float(self._consume()))
        if token.isalnum() or '_' in token or '.' in token:
            self._consume()
            if self._peek() == '(':
                self._consume()
                args = []
                if self._peek() != ')':
                    while True:
                        args.append(self._parse_expression())
                        if self._peek() == ')':
                            break
                        if self._peek() != ',':
                            raise ValueError("함수 인자 사이에 콤마(,)가 필요합니다.")
                        self._consume()
                return OperatorNode(token, args)
            else:
                return VariableNode(token)
        if token == '(':
            self._consume()
            expr = self._parse_expression()
            if self._consume() != ')':
                raise ValueError("괄호가 닫히지 않았습니다.")
            return expr
        raise ValueError(f"예상치 못한 토큰입니다: {token}")

    def _parse_binary_op(self, parse_next_level, ops: List[str]) -> ASTNode:
        node = parse_next_level()
        while self._peek() in ops:
            op = self._consume()
            right = parse_next_level()
            node = OperatorNode(op, [node, right])
        return node

    def _parse_power(self) -> ASTNode:
        return self._parse_binary_op(self._parse_primary, ['^'])

    def _parse_multiplicative(self) -> ASTNode:
        return self._parse_binary_op(self._parse_power, ['*', '/'])

    def _parse_additive(self) -> ASTNode:
        return self._parse_binary_op(self._parse_multiplicative, ['+', '-'])
        
    def _parse_comparison(self) -> ASTNode:
        return self._parse_binary_op(self._parse_additive, ['>', '<', '>=', '<=', '==', '!='])

    def _parse_logical(self) -> ASTNode:
        return self._parse_binary_op(self._parse_comparison, ['&&', '||'])

    def _parse_expression(self) -> ASTNode:
        node = self._parse_logical()
        if self._peek() == '?':
            self._consume()
            true_expr = self._parse_expression()
            if self._consume() != ':':
                raise ValueError("삼항 연산자에 ':'가 필요합니다.")
            false_expr = self._parse_expression()
            return OperatorNode('if', [node, true_expr, false_expr])
        return node

# --- 팩터 복잡도 분석기 (이전 코드와 동일) ---
class ComplexityAnalyzer:
    def calculate_symbolic_length(self, node: ASTNode) -> int:
        if isinstance(node, OperatorNode):
            return 1 + sum(self.calculate_symbolic_length(child) for child in node.children)
        return 1

    def calculate_parameter_count(self, node: ASTNode) -> int:
        count = 0
        if isinstance(node, LiteralNode) and isinstance(node.value, (int, float)):
            count = 1
        if isinstance(node, OperatorNode):
            count += sum(self.calculate_parameter_count(child) for child in node.children)
        return count


# --- 팩터 독창성 분석기 (수정됨) ---
class OriginalityAnalyzer:
    def __init__(self, parser: FactorParser, complexity_analyzer: ComplexityAnalyzer):
        self.parser = parser
        self.complexity_analyzer = complexity_analyzer
        self.alpha_zoo_asts = self._load_alpha_zoo_asts()

    def _load_alpha_zoo_asts(self) -> List[ASTNode]:
        formulas = get_alpha_zoo()
        asts = []
        for formula in formulas:
            try:
                asts.append(self.parser.parse(formula))
            except Exception as e:
                pass
        return asts

    def _get_all_subtrees(self, node: ASTNode) -> Set[ASTNode]:
        subtrees = {node}
        if isinstance(node, OperatorNode):
            for child in node.children:
                subtrees.update(self._get_all_subtrees(child))
        return subtrees

    def calculate_similarity_score(self, new_factor_ast: ASTNode) -> float:
        max_similarity = 0.0
        new_factor_subtrees = self._get_all_subtrees(new_factor_ast)
        size_new = self.complexity_analyzer.calculate_symbolic_length(new_factor_ast)

        for zoo_ast in self.alpha_zoo_asts:
            zoo_subtrees = self._get_all_subtrees(zoo_ast)
            
            common_subtrees = new_factor_subtrees.intersection(zoo_subtrees)
            
            if not common_subtrees:
                continue
            
            largest_common_subtree_size = 0
            for sub in common_subtrees:
                size = self.complexity_analyzer.calculate_symbolic_length(sub)
                if size > largest_common_subtree_size:
                    largest_common_subtree_size = size

            # 💡 논문 Eq.5에 따라 유사도 점수 계산 로직을 수정
            size_zoo = self.complexity_analyzer.calculate_symbolic_length(zoo_ast)
            if max(size_new, size_zoo) > 0:
                similarity = largest_common_subtree_size / max(size_new, size_zoo)
            else:
                similarity = 0.0

            if similarity > max_similarity:
                max_similarity = similarity
        
        return max_similarity
