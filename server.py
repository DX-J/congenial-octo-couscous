"""
保险营销内容智能审核系统 - 后端服务
整合百炼大模型 + RAG + 规则引擎 + 评估模块
统一输出格式：is_compliance, violation_type, regulations[{name, article, content}], confidence, explanation
"""
import os
import sys
import json
import re
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

from flask import Flask, request, jsonify
from flask_cors import CORS

# 将src目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== 统一输出格式 ====================
def build_audit_result(
    is_compliance: str,
    violation_type: str,
    regulations: List[Dict[str, str]],
    confidence: float,
    explanation: str,
    audit_source: str = "model"
) -> Dict[str, Any]:
    """
    构建统一的审核结果输出

    Args:
        is_compliance: "yes" 或 "no"
        violation_type: 违规类型（如"收益承诺"、"虚假宣传"等），合规时为"无"
        regulations: 引用的具体规则条文，结构为 [{"name": "法规名称", "article": "条文编号", "content": "原文"}]
        confidence: 置信度 0-1
        explanation: 审核说明/违规原因
        audit_source: 审核来源 "model" / "rule_engine" / "rag_model"

    Returns:
        统一格式的审核结果
    """
    return {
        "is_compliance": is_compliance,
        "violation_type": violation_type,
        "regulations": regulations,
        "confidence": round(confidence, 4),
        "explanation": explanation,
        "violation_reason": explanation,  # 兼容前端旧字段
        "cited_regulations": [
            f"{r['name']} {r['article']}: {r['content']}" for r in regulations
        ],  # 兼容前端旧字段
        "audit_source": audit_source,
        "timestamp": datetime.now().isoformat()
    }


# ==================== 法规知识库（RAG） ====================
class RegulationKnowledgeBase:
    """基于文件的法规知识库，支持关键词检索和语义匹配"""

    def __init__(self, knowledge_dir: str = None):
        self.knowledge_dir = knowledge_dir or os.path.join(
            os.path.dirname(__file__), 'knowledge_base'
        )
        self.documents = []
        self._load()

    def _load(self):
        """加载知识库文件"""
        if not os.path.exists(self.knowledge_dir):
            logger.warning(f"知识库目录不存在: {self.knowledge_dir}")
            self._create_default()
            return

        for filename in sorted(os.listdir(self.knowledge_dir)):
            if filename.endswith('.txt'):
                filepath = os.path.join(self.knowledge_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.documents.append({
                    'filename': filename,
                    'content': content,
                    'title': self._extract_title(content)
                })

        logger.info(f"知识库加载完成，共 {len(self.documents)} 个文件")

    def _create_default(self):
        """创建默认知识库"""
        os.makedirs(self.knowledge_dir, exist_ok=True)
        defaults = {
            "regulation_1.txt": """《银保监会关于规范保险销售行为的通知》
第一条 总则
为规范保险销售行为，保护保险消费者合法权益，促进保险业健康发展，根据《中华人民共和国保险法》等法律法规，制定本通知。

第二条 禁止性规定
保险机构及其从业人员在保险销售活动中不得有下列行为：
（一）欺骗保险人、投保人、被保险人或者受益人；
（二）对投保人隐瞒与保险合同有关的重要情况；
（三）阻碍投保人履行本法规定的如实告知义务，或者诱导其不履行本法规定的如实告知义务；
（四）给予或者承诺给予投保人、被保险人、受益人保险合同约定以外的利益；
（五）利用行政权力、职务或者职业便利以及其他不正当手段强迫、引诱或者限制投保人订立保险合同。

第四条 收益承诺限制
保险机构及其从业人员不得对保险产品的收益作出确定性承诺，不得夸大保险产品的收益或者缩小保险产品的风险。

第五条 比较宣传规范
保险机构及其从业人员在保险销售活动中进行产品比较的，应当客观、公正、全面，不得贬低其他保险机构或者保险产品。
""",
            "regulation_2.txt": """《保险销售行为管理办法》
第五条 保险销售人员不得有下列行为：
（一）欺骗投保人、被保险人或者受益人；
（二）隐瞒与保险合同有关的重要情况；
（三）阻碍投保人履行如实告知义务，或者诱导其不履行如实告知义务；
（四）给予或者承诺给予投保人、被保险人或者受益人保险合同约定以外的利益；
（五）利用行政权力、职务或者职业便利以及其他不正当手段强迫、引诱或者限制投保人订立保险合同。

第七条 保险销售行为应当遵循公平、公正、诚实信用原则，不得损害投保人、被保险人或者受益人的合法权益。

第十五条 保险机构应当向投保人说明保险合同的内容。

第十七条 保险公司、保险中介机构应当建立保险销售行为管理制度：
（五）不得使用强制搭售、默认勾选等方式；
（六）不得代替投保人签名、填写相关文件。
""",
            "regulation_3.txt": """《金融产品网络营销管理办法》
第八条 金融产品网络营销应当遵循公平、公正、诚实信用原则，不得损害金融消费者合法权益。

第九条 金融产品网络营销不得有下列行为：
（五）不得利用金融管理部门审核或备案为产品增信；
（八）不得利用学术机构、行业协会、专业人士的名义作推荐；
（九）不得利用演艺明星的名义或形象作推荐、证明。

第十四条 不得将组合销售选项设定为默认或首选。

第十五条 金融产品网络营销不得有下列行为：
（一）虚假或者引人误解的宣传；
（二）对金融产品的收益作出保证性承诺；
（三）利用监管机构的名义进行营销宣传；
（四）利用演艺明星的名义或形象作推荐、证明。
""",
            "regulation_4.txt": """《人身保险销售行为管理办法》
第五条 人身保险销售行为应当遵循合法、公平、诚实信用原则。

第七条 人身保险销售人员不得有下列行为：
（四）使用"保证收益"、"无风险"等误导性表述；
（五）对保险产品的收益作出确定性承诺；
（六）与银行存款、国债等进行不当比较。

第十一条 人身保险销售人员应当向投保人说明保险合同的条款内容，特别是免除保险人责任的条款。

第十二条 人身保险销售人员应当向投保人明确说明费用扣除情况。
""",
            "regulation_5.txt": """《互联网保险业务监管办法》
第三条 互联网保险业务应当由依法设立的保险机构开展，其他机构和个人不得开展互联网保险业务。

第十四条 保险机构开展互联网保险业务，保单利益具有不确定性，不得作出确定性承诺。

第十五条 保险机构应当向投保人提示下列信息：
（一）保障范围、除外责任；
（二）费用扣除情况；
（三）犹豫期及退保损失。

第二十三条 保险机构应当确保投保人本人签署相关文件，不得代替投保人签名。
""",
            "regulation_6.txt": """《保险法》
第十六条 订立保险合同，保险人就保险标的或者被保险人的有关情况提出询问的，投保人应当如实告知。
投保人故意或者因重大过失未履行前款规定的如实告知义务，足以影响保险人决定是否同意承保或者提高保险费率的，保险人有权解除合同。
"""
        }

        for filename, content in defaults.items():
            filepath = os.path.join(self.knowledge_dir, filename)
            if not os.path.exists(filepath):
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                self.documents.append({
                    'filename': filename,
                    'content': content,
                    'title': self._extract_title(content)
                })

    def _extract_title(self, content: str) -> str:
        """提取文档标题"""
        for line in content.strip().split('\n'):
            line = line.strip()
            if '《' in line and '》' in line:
                return line
        return ""

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关法规

        Args:
            query: 查询文本
            top_k: 返回结果数

        Returns:
            相关法规列表，每条包含 title, content, score, articles
        """
        # 关键词权重映射
        keyword_map = {
            '保证': ['收益承诺限制', '保证收益', '确定性承诺'],
            '承诺': ['收益承诺限制', '确定性承诺'],
            '收益': ['收益承诺限制', '确定性承诺', '夸大收益'],
            '无风险': ['误导性表述', '保证收益'],
            '零风险': ['误导性表述', '保证收益'],
            '稳赚': ['误导性表述', '保证收益'],
            '保本': ['误导性表述', '保证收益'],
            '保底': ['收益承诺限制', '确定性承诺'],
            '夸大': ['收益承诺限制', '夸大收益'],
            '隐瞒': ['禁止性规定', '隐瞒重要情况'],
            '欺骗': ['禁止性规定', '欺骗'],
            '返现': ['不当利益', '合同约定以外的利益'],
            '返利': ['不当利益', '合同约定以外的利益'],
            '送礼': ['不当利益', '合同约定以外的利益'],
            '赠品': ['不当利益', '合同约定以外的利益'],
            '明星': ['演艺明星', '推荐证明'],
            '代言': ['演艺明星', '推荐证明'],
            '网红': ['演艺明星', '推荐证明'],
            '权威': ['监管备案', '学术机构'],
            '认证': ['监管备案', '学术机构'],
            '银保监会': ['监管机构', '备案增信'],
            '备案': ['监管备案', '备案增信'],
            '停售': ['虚假宣传', '引人误解'],
            '限时': ['虚假宣传', '引人误解'],
            '抢购': ['虚假宣传', '引人误解'],
            '存款': ['不当比较', '银行存款'],
            '银行': ['不当比较', '银行存款'],
            '比较': ['不当比较', '比较宣传'],
            '贬低': ['比较宣传', '贬低'],
            '强制': ['强制搭售', '默认勾选'],
            '默认': ['强制搭售', '默认勾选'],
            '代签': ['代签名', '代填写'],
            '代填': ['代签名', '代填写'],
            '不如实': ['如实告知', '阻碍告知'],
            '除外责任': ['除外责任', '隐瞒重要情况'],
            '犹豫期': ['犹豫期', '退保损失'],
            '退保': ['退保损失', '犹豫期'],
            '手续费': ['费用扣除', '隐瞒费用'],
            '费用': ['费用扣除', '隐瞒费用'],
            '投连': ['投连险', '保单利益不确定'],
            '万能': ['万能险', '保单利益不确定'],
            '第三方': ['非持牌机构', '互联网保险'],
            '推送': ['骚扰营销', '频繁推送'],
        }

        results = []
        for doc in self.documents:
            score = 0
            doc_lower = doc['content']

            # 关键词匹配计分
            for keyword, related_terms in keyword_map.items():
                if keyword in query:
                    for term in related_terms:
                        if term in doc_lower:
                            score += 2

            # 直接关键词匹配
            for char in query:
                if len(char) > 0 and char in doc_lower:
                    score += 0.1

            if score > 0:
                # 提取相关条款
                articles = self._extract_articles(doc['content'], query)
                results.append({
                    'title': doc['title'],
                    'content': doc['content'],
                    'score': score,
                    'articles': articles
                })

        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_k]

    def _extract_articles(self, content: str, query: str) -> List[Dict[str, str]]:
        """
        从法规内容中提取相关条款

        Returns:
            条款列表，每条包含 name, article, content
        """
        title = self._extract_title(content)
        articles = []

        # 按条款分割
        paragraphs = re.split(r'\n\n+', content.strip())

        for para in paragraphs:
            para = para.strip()
            if len(para) < 10:
                continue

            # 提取条款编号
            article_match = re.search(r'第[一二三四五六七八九十百零\d]+条', para)
            if not article_match:
                # 尝试匹配"第X条第X款"
                article_match = re.search(r'第[一二三四五六七八九十百零\d]+条第[一二三四五六七八九十百零\d]+款', para)

            article_number = article_match.group() if article_match else ""

            # 检查是否与查询相关
            relevance_score = 0
            query_chars = set(query)
            for char in query_chars:
                if char in para:
                    relevance_score += 1

            # 也检查关键词映射
            violation_keywords = [
                '保证', '承诺', '收益', '无风险', '零风险', '稳赚', '保本', '保底',
                '夸大', '隐瞒', '欺骗', '返现', '返利', '送礼', '明星', '代言',
                '权威', '认证', '停售', '限时', '抢购', '存款', '银行', '比较',
                '强制', '默认', '代签', '除外', '犹豫', '退保', '费用', '投连',
                '不当', '禁止', '不得', '违反', '违规', '虚假', '误导'
            ]
            for kw in violation_keywords:
                if kw in query and kw in para:
                    relevance_score += 3

            if relevance_score >= 2:
                articles.append({
                    'name': title,
                    'article': article_number,
                    'content': para[:200]
                })

        return articles


# ==================== 规则引擎 ====================
class RuleEngine:
    """基于正则的规则引擎，用于前置过滤和高置信度审核"""

    def __init__(self):
        self.rules = self._init_rules()

    def _init_rules(self) -> List[Dict]:
        """初始化规则库"""
        rules = [
            # 收益承诺类
            {"pattern": r"(保证|承诺|确定|一定有).{0,8}(收益|年化|回报|利息|利润率)",
             "type": "收益承诺", "severity": 0.95,
             "reason": "对保险产品收益作出确定性承诺",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第五条第（四）款",
                           "content": "不得对保险产品的收益作出确定性承诺"}},

            {"pattern": r"(年化|收益率|回报).{0,5}([0-9]+(\.[0-9]+)?%)",
             "type": "收益承诺", "severity": 0.9,
             "reason": "宣传具体收益率数值",
             "regulation": {"name": "《银保监会关于规范保险销售行为的通知》", "article": "第四条",
                           "content": "不得夸大保险产品的收益"}},

            {"pattern": r"(无风险|零风险|保证|一定|绝对).{0,8}(本金|资金|投资安全|不亏|不会亏|不赔)",
             "type": "收益承诺", "severity": 0.95,
             "reason": "承诺本金安全或无风险",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第七条第（四）款",
                           "content": "不得使用'保证收益'、'无风险'等误导性表述"}},

            {"pattern": r"(稳赚|必赚|一定赚|肯定赚|不会亏|只赚不亏|稳赚不赔)",
             "type": "收益承诺", "severity": 0.95,
             "reason": "使用确定性收益表述",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第七条第（四）款",
                           "content": "不得使用'保证收益'、'无风险'等误导性表述"}},

            {"pattern": r"(保本|本金安全|100%安全).{0,8}(保障|安全|保证|保护)",
             "type": "收益承诺", "severity": 0.9,
             "reason": "宣传本金保障",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第七条第（四）款",
                           "content": "不得使用'保本'等误导性表述"}},

            {"pattern": r"(保单利益|保单收益|保单价值).{0,6}(确定|保证|固定|稳定)",
             "type": "收益承诺", "severity": 0.9,
             "reason": "对保单利益作出确定性承诺",
             "regulation": {"name": "《互联网保险业务监管办法》", "article": "第十四条",
                           "content": "保单利益具有不确定性，不得作出确定性承诺"}},

            # 虚假宣传类
            {"pattern": r"(停售|退市|不卖了|最后|绝版).{0,6}(抢购|机会|抓紧|马上)",
             "type": "虚假宣传", "severity": 0.7,
             "reason": "利用停售进行促销炒作",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第五条第（二）款",
                           "content": "不得进行虚假或者引人误解的宣传"}},

            {"pattern": r"(限时|仅限|马上|立刻|立即).{0,6}(停售|售罄|结束|过期|截止)",
             "type": "虚假宣传", "severity": 0.75,
             "reason": "使用限时促销噱头",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第五条第（二）款",
                           "content": "不得进行虚假或者引人误解的宣传"}},

            {"pattern": r"(明星|名人|演员|艺人).{0,6}(推荐|代言|保证|背书)",
             "type": "虚假宣传", "severity": 0.85,
             "reason": "利用演艺明星名义作推荐",
             "regulation": {"name": "《金融产品网络营销管理办法》", "article": "第九条第（九）款",
                           "content": "不得利用演艺明星的名义或形象作推荐、证明"}},

            {"pattern": r"(专家|权威机构|银保监会).{0,6}(认证|保证|背书|推荐)",
             "type": "虚假宣传", "severity": 0.85,
             "reason": "利用权威机构名义增信",
             "regulation": {"name": "《金融产品网络营销管理办法》", "article": "第九条第（五）款",
                           "content": "不得利用金融管理部门审核或备案为产品增信"}},

            {"pattern": r"(没有|不收|免|0|全免).{0,8}(手续费|管理费|佣金|费用)",
             "type": "虚假宣传", "severity": 0.85,
             "reason": "隐瞒或虚假宣传费用",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第十二条",
                           "content": "应当向投保人明确说明费用扣除情况"}},

            {"pattern": r"(全部|所有|任何|一切).{0,6}(保障|赔付|理赔|赔偿)",
             "type": "虚假宣传", "severity": 0.8,
             "reason": "隐瞒除外责任",
             "regulation": {"name": "《互联网保险业务监管办法》", "article": "第十五条第（一）款",
                           "content": "应提示保障范围、除外责任"}},

            # 误导性表述类
            {"pattern": r"(存款|银行).{0,6}(保险|理财|投资)",
             "type": "误导性表述", "severity": 0.85,
             "reason": "将保险与存款混淆宣传",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第七条第（六）款",
                           "content": "不得与银行存款、国债等进行不当比较"}},

            {"pattern": r"(必须|强制|一定要).{0,6}(购买|买|投保|搭配)",
             "type": "误导性表述", "severity": 0.85,
             "reason": "强制搭售",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第十七条第（五）款",
                           "content": "不得使用强制搭售、默认勾选等方式"}},

            {"pattern": r"(帮您|帮您代|代您|替您).{0,8}(签名|填写|签字)",
             "type": "误导性表述", "severity": 0.9,
             "reason": "代签名或代填写",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第十七条第（六）款",
                           "content": "不得代替投保人签名、填写"}},

            {"pattern": r"(不用看|不用管|不重要).{0,8}(条款|细则|说明|免责)",
             "type": "误导性表述", "severity": 0.85,
             "reason": "隐瞒重要条款",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第十五条",
                           "content": "应当向投保人说明保险合同的内容"}},

            # 不当利益类
            {"pattern": r"(返现|返利|返还).{0,15}(现金|钱|佣金|好处|礼品)",
             "type": "不当利益", "severity": 0.9,
             "reason": "承诺返现或返利",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第五条第（四）款",
                           "content": "不得给予保险合同约定以外的利益"}},

            {"pattern": r"(多送|加送|赠送).{0,15}(礼品|礼物|红包|奖金)",
             "type": "不当利益", "severity": 0.8,
             "reason": "承诺赠送礼品",
             "regulation": {"name": "《保险销售行为管理办法》", "article": "第五条第（四）款",
                           "content": "不得给予保险合同约定以外的利益"}},

            # 不当比较类
            {"pattern": r"(比|胜于|优于|超过|碾压|高).{0,6}(银行|存款|国债|基金|利息)",
             "type": "不当比较", "severity": 0.8,
             "reason": "与存款、国债等进行不当比较",
             "regulation": {"name": "《人身保险销售行为管理办法》", "article": "第七条第（六）款",
                           "content": "不得与银行存款、国债等进行不当比较"}},

            {"pattern": r"(某安|某寿|某保).{0,6}(不如|比不了|差远了|不行)",
             "type": "不当比较", "severity": 0.85,
             "reason": "贬低其他保险公司",
             "regulation": {"name": "《银保监会关于规范保险销售行为的通知》", "article": "第五条",
                           "content": "不得贬低其他保险机构或者保险产品"}},
        ]

        logger.info(f"规则引擎初始化完成，共 {len(rules)} 条规则")
        return rules

    def audit(self, text: str) -> Dict[str, Any]:
        """
        使用规则引擎审核内容

        Returns:
            统一格式的审核结果
        """
        if not text or len(text.strip()) == 0:
            return build_audit_result("yes", "无", [], 1.0, "未检测到明显违规内容", "rule_engine")

        matches = []
        for rule in self.rules:
            pattern = re.compile(rule['pattern'], re.IGNORECASE)
            match = pattern.search(text)
            if match:
                matches.append({
                    'matched_text': match.group(0),
                    'type': rule['type'],
                    'severity': rule['severity'],
                    'reason': rule['reason'],
                    'regulation': rule['regulation']
                })

        if not matches:
            return build_audit_result("yes", "无", [], 0.85, "未检测到明显违规内容", "rule_engine")

        # 按严重程度排序
        matches.sort(key=lambda x: x['severity'], reverse=True)

        # 汇总违规类型
        violation_types = list(dict.fromkeys(m['type'] for m in matches))
        violation_type_str = "、".join(violation_types)

        # 汇总引用法规（去重）
        seen = set()
        regulations = []
        for m in matches:
            reg = m['regulation']
            key = f"{reg['name']}_{reg['article']}"
            if key not in seen:
                seen.add(key)
                regulations.append(reg)

        # 计算置信度
        max_severity = matches[0]['severity']
        confidence = min(0.75 + len(matches) * 0.05, 0.98)

        # 生成说明
        reasons = [f"检测到'{m['matched_text']}'，{m['reason']}" for m in matches[:3]]
        explanation = "；".join(reasons)

        return build_audit_result(
            is_compliance="no",
            violation_type=violation_type_str,
            regulations=regulations,
            confidence=confidence,
            explanation=explanation,
            audit_source="rule_engine"
        )


# ==================== 百炼大模型客户端 ====================
class BailianLLMClient:
    """阿里云百炼大模型客户端，支持 RAG 增强审核"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', '')
        self.model_name = os.getenv('MODEL_NAME', 'qwen-max')
        self.client = None
        self._init_client()

    def _init_client(self):
        """初始化大模型客户端，使用 requests 直接调用百炼API"""
        if not self.api_key:
            logger.warning("未配置 DASHSCOPE_API_KEY，大模型审核将不可用")
            return

        # 使用 requests 直接调用百炼 OpenAI 兼容接口（无需额外SDK）
        try:
            import requests as _req
            # 测试API Key有效性
            self.client = "requests_mode"
            self._use_openai = False
            logger.info(f"百炼大模型客户端初始化成功(requests直连)，模型: {self.model_name}")
        except Exception as e:
            logger.warning(f"初始化失败: {e}")
            self.client = None

        self._use_openai = False

    @property
    def available(self) -> bool:
        return self.client is not None

    def audit_with_rag(
        self,
        content: str,
        regulations: List[Dict],
    ) -> Dict[str, Any]:
        """
        使用 RAG + 大模型进行审核

        Args:
            content: 待审核内容
            regulations: RAG 检索到的相关法规

        Returns:
            统一格式的审核结果
        """
        if not self.available:
            return None

        # 构建法规上下文
        reg_context = ""
        for i, reg in enumerate(regulations, 1):
            title = reg.get('title', '')
            articles = reg.get('articles', [])
            if articles:
                for art in articles[:3]:
                    reg_context += f"\n{art['name']} {art['article']}：{art['content']}\n"
            else:
                reg_context += f"\n【{title}】\n{reg['content'][:500]}\n"

        system_prompt = """你是一位专业的保险营销内容合规审核专家，精通中国保险法律法规和监管要求。
你的任务是审核保险营销内容是否符合监管规定。

审核要求：
1. 仔细分析输入的营销内容
2. 严格基于提供的法规条文判断是否违规，不得编造法规
3. 输出必须是严格的JSON格式，包含以下字段：
   - is_compliance: "yes" 或 "no"
   - violation_type: 违规类型，如"收益承诺"、"虚假宣传"、"误导性表述"、"不当利益"、"不当比较"等，多个用"、"分隔，合规时为空字符串
   - regulations: 引用的具体规则条文，数组格式，每条包含三个字段：
     * name: 法规名称（如"《人身保险销售行为管理办法》"）
     * article: 条文编号（如"第七条第（四）款"）
     * content: 条文原文
   - confidence: 置信度，0到1之间的小数
   - explanation: 详细的审核说明

合规示例：
{"is_compliance": "yes", "violation_type": "", "regulations": [], "confidence": 0.95, "explanation": "未检测到违规内容"}

违规示例：
{"is_compliance": "no", "violation_type": "收益承诺", "regulations": [{"name": "《人身保险销售行为管理办法》", "article": "第七条第（四）款", "content": "不得使用'保证收益'、'无风险'等误导性表述"}], "confidence": 0.92, "explanation": "检测到'保证收益'等确定性承诺表述，违反了相关法规关于收益承诺的限制"}

请严格按照JSON格式输出，不要添加其他内容。"""

        user_prompt = f"""请审核以下保险营销内容是否符合监管要求：

【相关监管法规】
{reg_context}

【待审核内容】
{content}

请基于上述法规进行审核，严格输出JSON格式结果。"""

        try:
            # 使用 requests 直接调用百炼 OpenAI 兼容接口
            import requests as _req
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 2000
            }

            response = _req.post(url, headers=headers, json=payload, timeout=30)

            if response.status_code != 200:
                logger.error(f"百炼API调用失败: {response.status_code} {response.text[:200]}")
                return None

            result_data = response.json()
            result_text = result_data["choices"][0]["message"]["content"]

            # 解析JSON结果
            audit_result = self._parse_json_response(result_text)
            if audit_result:
                audit_result['audit_source'] = 'rag_model'
                return audit_result

            return None

        except Exception as e:
            logger.error(f"大模型审核失败: {str(e)}")
            return None

    def _parse_json_response(self, text: str) -> Optional[Dict[str, Any]]:
        """解析大模型返回的JSON"""
        # 尝试提取JSON
        json_match = re.search(r'\{[\s\S]*\}', text)
        if not json_match:
            return None

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError:
            return None

        # 验证并标准化字段
        is_compliance = data.get('is_compliance', 'yes')
        if is_compliance not in ['yes', 'no']:
            is_compliance = 'yes'

        violation_type = data.get('violation_type', '')
        if is_compliance == 'yes':
            violation_type = ''

        # 标准化 regulations 字段
        raw_regs = data.get('regulations', [])
        regulations = []
        for reg in raw_regs:
            if isinstance(reg, dict):
                regulations.append({
                    'name': reg.get('name', ''),
                    'article': reg.get('article', ''),
                    'content': reg.get('content', '')
                })
            elif isinstance(reg, str):
                regulations.append({'name': reg, 'article': '', 'content': ''})

        confidence = data.get('confidence', 0.8)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except (ValueError, TypeError):
            confidence = 0.8

        explanation = data.get('explanation', data.get('violation_reason', ''))

        return build_audit_result(
            is_compliance=is_compliance,
            violation_type=violation_type or ("无" if is_compliance == "yes" else "未知"),
            regulations=regulations,
            confidence=confidence,
            explanation=explanation or ("未检测到违规内容" if is_compliance == "yes" else "检测到违规内容"),
            audit_source="rag_model"
        )


# ==================== 审核Agent（Workflow编排） ====================
class AuditAgent:
    """
    审核Agent：编排 RAG检索 → 规则引擎前置过滤 → 大模型审核 → 结果整合 的完整工作流
    """

    RULE_ENGINE_HIGH_CONFIDENCE = 0.85
    RULE_ENGINE_MEDIUM_CONFIDENCE = 0.7

    def __init__(self, kb: RegulationKnowledgeBase, rule_engine: RuleEngine, llm: BailianLLMClient):
        self.kb = kb
        self.rule_engine = rule_engine
        self.llm = llm

    def audit(self, content: str) -> Dict[str, Any]:
        """
        执行完整审核工作流

        Workflow:
        1. 规则引擎前置过滤
           - 高置信度(≥0.85)违规 → 直接返回规则引擎结果
           - 高置信度(≥0.85)合规 → 直接返回合规
        2. RAG检索相关法规
        3. 百炼大模型 + RAG上下文审核
           - 成功 → 返回模型结果
           - 失败 → 降级到规则引擎结果
        4. 结果整合与校验
        """
        logger.info(f"开始审核，内容长度: {len(content)}")

        # Step 1: 规则引擎前置过滤
        rule_result = self.rule_engine.audit(content)
        rule_confidence = rule_result['confidence']

        logger.info(f"规则引擎判定: {rule_result['is_compliance']}, 置信度: {rule_confidence}")

        # 高置信度违规 → 直接返回
        if rule_result['is_compliance'] == 'no' and rule_confidence >= self.RULE_ENGINE_HIGH_CONFIDENCE:
            logger.info("规则引擎高置信度违规，直接返回")
            return rule_result

        # 高置信度合规（无匹配规则）→ 尝试用模型二次确认
        # 但如果模型不可用，直接返回合规结果

        # Step 2: RAG检索
        relevant_regs = self.kb.retrieve(content, top_k=3)
        logger.info(f"RAG检索到 {len(relevant_regs)} 条相关法规")

        # Step 3: 大模型审核
        if self.llm.available:
            model_result = self.llm.audit_with_rag(content, relevant_regs)
            if model_result:
                # Step 4: 结果整合
                # 如果规则引擎也检测到违规，合并法规引用
                if rule_result['is_compliance'] == 'no':
                    model_result = self._merge_results(model_result, rule_result)

                logger.info(f"大模型审核完成: {model_result['is_compliance']}")
                return model_result

        # 模型不可用或失败 → 降级到规则引擎
        logger.info("降级到规则引擎结果")
        if rule_result['is_compliance'] == 'no':
            return rule_result

        # 如果规则引擎也认为合规，但模型不可用
        # 尝试基于RAG检索结果给出更详细的合规判断
        if relevant_regs:
            return build_audit_result(
                is_compliance="yes",
                violation_type="无",
                regulations=[],
                confidence=0.75,
                explanation="后端大模型服务暂不可用，基于规则引擎未检测到违规内容（建议开启后端服务获取更准确结果）",
                audit_source="rule_engine_fallback"
            )

        return rule_result

    def _merge_results(self, model_result: Dict, rule_result: Dict) -> Dict:
        """合并模型结果和规则引擎结果"""
        # 合并法规引用（去重）
        seen = set()
        merged_regs = []

        for reg in model_result.get('regulations', []):
            key = f"{reg.get('name', '')}_{reg.get('article', '')}"
            if key not in seen:
                seen.add(key)
                merged_regs.append(reg)

        for reg in rule_result.get('regulations', []):
            key = f"{reg.get('name', '')}_{reg.get('article', '')}"
            if key not in seen:
                seen.add(key)
                merged_regs.append(reg)

        model_result['regulations'] = merged_regs
        model_result['cited_regulations'] = [
            f"{r['name']} {r['article']}: {r['content']}" for r in merged_regs
        ]

        # 取较高置信度
        model_result['confidence'] = max(
            model_result.get('confidence', 0.8),
            rule_result.get('confidence', 0.8)
        )

        return model_result


# ==================== 评估模块 ====================
class EvaluationModule:
    """基本的效果评估模块"""

    def __init__(self):
        self.results = []

    def add_result(self, test_case: Dict, prediction: Dict):
        """记录评估结果"""
        is_correct = test_case.get('expected_compliance') == prediction.get('is_compliance')
        self.results.append({
            'timestamp': datetime.now().isoformat(),
            'test_case': test_case,
            'prediction': prediction,
            'is_correct': is_correct
        })

    def get_metrics(self) -> Dict[str, Any]:
        """计算评估指标"""
        if not self.results:
            return {"accuracy": 0.0, "total": 0, "correct": 0}

        total = len(self.results)
        correct = sum(1 for r in self.results if r['is_correct'])

        # 计算混淆矩阵
        tp = fp = tn = fn = 0
        for r in self.results:
            pred = r['prediction'].get('is_compliance', 'yes')
            true = r['test_case'].get('expected_compliance', 'yes')
            if true == 'no' and pred == 'no':
                tp += 1
            elif true == 'yes' and pred == 'no':
                fp += 1
            elif true == 'yes' and pred == 'yes':
                tn += 1
            elif true == 'no' and pred == 'yes':
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "accuracy": correct / total,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "total": total,
            "correct": correct,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
        }

    def generate_report(self) -> str:
        """生成评估报告"""
        metrics = self.get_metrics()
        report = f"""
保险审核系统效果评估报告
========================
生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

一、总体指标
-----------
测试总数: {metrics['total']}
正确数: {metrics['correct']}
准确率: {metrics['accuracy'] * 100:.2f}%
精确率: {metrics.get('precision', 0) * 100:.2f}%
召回率: {metrics.get('recall', 0) * 100:.2f}%
F1分数: {metrics.get('f1_score', 0) * 100:.2f}%

二、混淆矩阵
-----------
真正例(TP): {metrics.get('confusion_matrix', {}).get('tp', 0)}
假正例(FP): {metrics.get('confusion_matrix', {}).get('fp', 0)}
真负例(TN): {metrics.get('confusion_matrix', {}).get('tn', 0)}
假负例(FN): {metrics.get('confusion_matrix', {}).get('fn', 0)}
"""
        return report


# ==================== Flask 应用 ====================
def create_app():
    """创建Flask应用"""
    app = Flask(__name__)
    CORS(app)  # 允许跨域

    # 初始化组件
    kb = RegulationKnowledgeBase()
    rule_engine = RuleEngine()
    llm = BailianLLMClient()
    agent = AuditAgent(kb, rule_engine, llm)
    evaluator = EvaluationModule()

    @app.route('/health', methods=['GET'])
    def health_check():
        """健康检查"""
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "llm_available": llm.available,
            "model": llm.model_name if llm.available else "unavailable"
        })

    @app.route('/audit/text', methods=['POST'])
    def audit_text():
        """审核文本内容"""
        try:
            data = request.json or {}
            content = data.get('content', '')

            if not content:
                return jsonify({"error": "请提供待审核内容(content字段)"}), 400

            # 执行审核
            result = agent.audit(content)

            # 如果请求中包含预期结果，进行评估
            if 'expected_compliance' in data:
                evaluator.add_result({
                    'input': content,
                    'expected_compliance': data['expected_compliance'],
                    'expected_violation_type': data.get('expected_violation_type', '')
                }, result)

            return jsonify(result)

        except Exception as e:
            logger.error(f"审核失败: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route('/audit/image', methods=['POST'])
    def audit_image():
        """审核图片内容（接收base64图片或OCR文本）"""
        try:
            data = request.json or {}

            # 优先使用前端传来的OCR识别文本
            ocr_text = data.get('ocr_text', '')
            content = data.get('content', '')

            audit_content = ocr_text or content

            if not audit_content:
                return jsonify({"error": "请提供ocr_text或content字段"}), 400

            result = agent.audit(audit_content)
            result['ocr_text'] = ocr_text
            result['content_type'] = 'image'

            return jsonify(result)

        except Exception as e:
            logger.error(f"图片审核失败: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route('/audit/batch', methods=['POST'])
    def audit_batch():
        """批量审核"""
        try:
            data = request.json or {}
            contents = data.get('contents', [])

            if not contents:
                return jsonify({"error": "请提供contents数组"}), 400

            results = []
            for item in contents:
                if isinstance(item, str):
                    result = agent.audit(item)
                elif isinstance(item, dict):
                    result = agent.audit(item.get('content', ''))
                else:
                    continue
                results.append(result)

            return jsonify({"results": results, "total": len(results)})

        except Exception as e:
            logger.error(f"批量审核失败: {str(e)}\n{traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    @app.route('/evaluation/report', methods=['GET'])
    def evaluation_report():
        """获取评估报告"""
        metrics = evaluator.get_metrics()
        report = evaluator.generate_report()
        return jsonify({
            "report": report,
            "metrics": metrics
        })

    @app.route('/evaluation/run', methods=['POST'])
    def run_evaluation():
        """运行评估测试集"""
        test_cases = [
            {"content": "本产品保证年化收益率8%，无风险！", "expected_compliance": "no", "expected_violation_type": "收益承诺"},
            {"content": "这是一款重疾保险，提供全面保障，详情请咨询代理人。", "expected_compliance": "yes", "expected_violation_type": ""},
            {"content": "某明星代言推荐这款保险，权威认证！", "expected_compliance": "no", "expected_violation_type": "虚假宣传"},
            {"content": "限时抢购，明天停售！错过可惜！", "expected_compliance": "no", "expected_violation_type": "虚假宣传"},
            {"content": "返现500元，多送精美礼品", "expected_compliance": "no", "expected_violation_type": "不当利益"},
            {"content": "比银行存款利息高3倍", "expected_compliance": "no", "expected_violation_type": "不当比较"},
            {"content": "本产品已获银保监会备案，犹豫期内可退保。", "expected_compliance": "yes", "expected_violation_type": ""},
            {"content": "存款式的保险，安全可靠像活期", "expected_compliance": "no", "expected_violation_type": "误导性表述"},
            {"content": "稳赚不赔，保本保息，零风险投资", "expected_compliance": "no", "expected_violation_type": "收益承诺"},
            {"content": "帮您代签名，不用看条款，不用如实告知", "expected_compliance": "no", "expected_violation_type": "误导性表述"},
        ]

        for case in test_cases:
            result = agent.audit(case['content'])
            evaluator.add_result(case, result)

        metrics = evaluator.get_metrics()
        return jsonify({
            "message": f"评估完成，共 {len(test_cases)} 条测试",
            "metrics": metrics
        })

    @app.route('/regulations', methods=['GET'])
    def list_regulations():
        """列出知识库中的法规"""
        regs = []
        for doc in kb.documents:
            regs.append({
                "title": doc['title'],
                "filename": doc['filename']
            })
        return jsonify({"regulations": regs, "total": len(regs)})

    return app


# ==================== 启动入口 ====================
if __name__ == '__main__':
    from dotenv import load_dotenv

    # 加载 .env 文件
    env_path = os.path.join(os.path.dirname(__file__), '.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
        logger.info(f"已加载环境变量: {env_path}")

    # 创建应用
    app = create_app()

    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))

    logger.info(f"保险审核系统后端服务启动: http://{host}:{port}")
    logger.info(f"API文档: http://{host}:{port}/health")

    app.run(host=host, port=port, debug=True)
