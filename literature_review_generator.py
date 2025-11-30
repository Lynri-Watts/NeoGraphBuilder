#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于知识图谱的文献综述生成脚本

该脚本实现多Agent协作的文献综述生成：
1. SearchAgent：反复查找相关节点和路径，形成多条解决问题的路径
2. GeneratorAgent：基于查找结果生成最终文献综述
3. 支持文献信息的unicode解码和结构化展示
"""

import os
import json
import logging
import time
import re
import ast
import datetime
import subprocess
import concurrent.futures
from collections import defaultdict
from typing import List, Dict, Any, Set
from openai import OpenAI

# 导入数据库模块和LLM客户端
from database import Neo4jKnowledgeGraph
from llm import LLMClient
from document_manager import DocumentManager

# 配置日志
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ReportGenerator:
    """
    报告生成器类，负责收集和保存每个Agent的输出内容
    """
    
    def __init__(self, output_dir="reports"):
        """
        初始化报告生成器
        
        Args:
            output_dir: 报告输出目录
        """
        self.output_dir = output_dir
        self.session_data = {
            "session_info": {},
            "agents_output": {},
            "errors": []
        }
        self.session_start_time = None
        self.agent_start_times = {}
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def start_session(self, topic):
        """
        开始一个新的报告会话
        
        Args:
            topic: 研究主题
        """
        self.session_start_time = time.time()
        self.session_data = {
            "session_info": {
                "topic": topic,
                "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "timestamp": self.session_start_time,
                "total_duration": 0
            },
            "agents_output": {},
            "errors": []
        }
        self.agent_start_times = {}
        logger.info(f"开始报告会话，主题: {topic}")
    
    def record_agent_start(self, agent_name):
        """
        记录Agent开始工作
        
        Args:
            agent_name: Agent名称
        """
        self.agent_start_times[agent_name] = time.time()
        self.session_data["agents_output"][agent_name] = {
            "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": None,
            "duration": None,
            "output_data": None,
            "success": False
        }
        logger.info(f"记录Agent开始工作: {agent_name}")
    
    def record_agent_end(self, agent_name, output_data=None):
        """
        记录Agent结束工作并保存输出数据
        
        Args:
            agent_name: Agent名称
            output_data: Agent的输出数据
        """
        if agent_name in self.agent_start_times:
            duration = time.time() - self.agent_start_times[agent_name]
            end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.session_data["agents_output"][agent_name].update({
                "end_time": end_time,
                "duration": duration,
                "output_data": output_data,
                "success": True
            })
            
            logger.info(f"记录Agent完成工作: {agent_name}, 耗时: {duration:.2f}秒")
        else:
            logger.warning(f"Agent {agent_name} 没有开始记录，无法记录结束")
    
    def record_error(self, error_message, agent_name=None):
        """
        记录错误信息
        
        Args:
            error_message: 错误信息
            agent_name: 相关的Agent名称（可选）
        """
        error_data = {
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "agent": agent_name,
            "message": error_message
        }
        self.session_data["errors"].append(error_data)
        logger.error(f"记录错误: {error_message}")
    
    def end_session(self):
        """结束当前会话"""
        if self.session_start_time:
            total_duration = time.time() - self.session_start_time
            self.session_data["session_info"]["total_duration"] = total_duration
            self.session_data["session_info"]["end_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            logger.info(f"报告会话结束，总耗时: {total_duration:.2f}秒")
    
    def save_report(self, stage=None):
        """
        保存报告到文件
        
        Args:
            stage: 报告阶段标识（可选）
            
        Returns:
            保存的文件路径
        """
        try:
            # 生成文件名
            topic_safe = "".join(c for c in self.session_data["session_info"]["topic"] if c.isalnum() or c in (' ', '-', '_')).rstrip()
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 根据阶段添加标识
            stage_suffix = f"_{stage}" if stage else ""
            filename = f"literature_review_report{stage_suffix}_{topic_safe}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存JSON格式的详细报告
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.session_data, f, ensure_ascii=False, indent=2)
            
            # 同时保存一个简化的Markdown报告
            markdown_filename = f"literature_review_summary{stage_suffix}_{topic_safe}_{timestamp}.md"
            markdown_filepath = os.path.join(self.output_dir, markdown_filename)
            self._save_markdown_report(markdown_filepath, stage)
            
            logger.info(f"{stage + ' ' if stage else ''}报告已保存到: {filepath}")
            return filepath
            
        except Exception as e:
            logger.error(f"保存报告时出错: {str(e)}")
            return None
    
    def save_stage_report(self, stage_name, stage_data=None):
        """
        保存阶段性报告
        
        Args:
            stage_name: 阶段名称
            stage_data: 阶段数据
            
        Returns:
            保存的文件路径
        """
        try:
            # 确保session_data中有stages字段
            if "stages" not in self.session_data:
                self.session_data["stages"] = {}
            
            # 记录阶段信息
            self.session_data["stages"][stage_name] = {
                "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "data": stage_data
            }
            
            # 保存阶段性报告
            return self.save_report(stage=stage_name)
            
        except Exception as e:
            logger.error(f"保存阶段性报告时出错: {str(e)}")
            return None
    
    def _save_markdown_report(self, filepath, stage=None):
        """
        保存Markdown格式的简化报告
        
        Args:
            filepath: Markdown文件路径
            stage: 当前阶段（可选）
        """
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                # 标题
                stage_title = f" - {stage}阶段" if stage else ""
                f.write(f"# 文献综述生成工作报告{stage_title}\n\n")
                
                # 会话信息
                session_info = self.session_data["session_info"]
                f.write("## 会话信息\n\n")
                f.write(f"- **研究主题**: {session_info['topic']}\n")
                f.write(f"- **开始时间**: {session_info['start_time']}\n")
                if 'end_time' in session_info:
                    f.write(f"- **结束时间**: {session_info['end_time']}\n")
                f.write(f"- **总耗时**: {session_info['total_duration']:.2f}秒\n\n")
                
                # 阶段信息
                if "stages" in self.session_data and self.session_data["stages"]:
                    f.write("## 阶段信息\n\n")
                    for stage_name, stage_data in self.session_data["stages"].items():
                        f.write(f"### {stage_name}\n\n")
                        f.write(f"- **时间**: {stage_data.get('timestamp', 'N/A')}\n")
                        # 根据不同阶段添加特定信息
                        if stage_name == "search_completed" and "data" in stage_data:
                            data = stage_data["data"]
                            if isinstance(data, dict) and 'search_paths' in data:
                                f.write(f"- **找到的路径数量**: {len(data['search_paths'])}\n")
                        elif stage_name == "documents_processed" and "data" in stage_data:
                            data = stage_data["data"]
                            if isinstance(data, dict) and 'processed_paths' in data:
                                f.write(f"- **处理的文献路径**: {len(data['processed_paths'])}\n")
                        elif stage_name == "review_generated" and "data" in stage_data:
                            data = stage_data["data"]
                            if isinstance(data, dict):
                                f.write(f"- **综述类型**: {'LaTeX' if data.get('use_latex', False) else '普通格式'}\n")
                        f.write(f"\n")
                
                # Agent工作情况
                f.write("## Agent工作情况\n\n")
                for agent_name, agent_data in self.session_data["agents_output"].items():
                    f.write(f"### {agent_name}\n\n")
                    f.write(f"- **开始时间**: {agent_data['start_time']}\n")
                    if agent_data['end_time']:
                        f.write(f"- **结束时间**: {agent_data['end_time']}\n")
                    f.write(f"- **耗时**: {agent_data['duration']:.2f}秒\n")
                    f.write(f"- **状态**: {'✅ 成功' if agent_data['success'] else '❌ 失败'}\n\n")
                
                # 错误信息
                if self.session_data["errors"]:
                    f.write("## 错误信息\n\n")
                    for error in self.session_data["errors"]:
                        f.write(f"- **时间**: {error['timestamp']}\n")
                        if error['agent']:
                            f.write(f"- **Agent**: {error['agent']}\n")
                        f.write(f"- **错误**: {error['message']}\n\n")
                
        except Exception as e:
            logger.error(f"保存Markdown报告时出错: {str(e)}")

# 从配置文件读取配置
import configparser
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# 数据库配置
NEO4J_CONFIG = {
    "uri": config.get('NEO4J_CONFIG', 'uri'),
    "user": config.get('NEO4J_CONFIG', 'user'),
    "password": config.get('NEO4J_CONFIG', 'password')
}

def decode_unicode_in_source_documents(source_documents):
    """
    解码source_documents中的unicode编码的title字段
    
    Args:
        source_documents: 包含unicode编码的source_documents列表
        
    Returns:
        解码后的source_documents列表
    """
    if not source_documents:
        return []
    
    decoded_documents = []
    for doc in source_documents:
        if isinstance(doc, str):
            try:
                # 尝试解析JSON字符串
                doc_dict = json.loads(doc)
                # JSON会自动处理unicode转义序列，不需要额外解码
                decoded_documents.append(doc_dict)
            except json.JSONDecodeError:
                # 如果解析失败，创建一个默认文档
                decoded_documents.append({"filename": doc, "title": doc, "paragraph_number": None})
        elif isinstance(doc, dict):
            # 如果已经是字典，直接使用
            decoded_documents.append(doc)
        else:
            decoded_documents.append({"filename": str(doc), "title": str(doc), "paragraph_number": None})
    
    return decoded_documents

class SearchAgent:
    """
    搜索Agent：负责从知识图谱中查找相关研究路径
    重构版：两步流程 - 1)提取关键词和节点 2)大模型决策扩展节点集合
    """
    
    def __init__(self, kg, llm_client=None, report_generator=None):
        """
        初始化SearchAgent
        
        Args:
            kg: 知识图谱实例
            llm_client: 大模型客户端实例
            report_generator: 报告生成器实例
        """
        self.kg = kg
        self.llm_client = llm_client
        self.report_generator = report_generator
        self.visited_nodes = set()
        self.current_node_set = {}  # 当前节点集合 {node_id: node_info}
        self.search_paths = []
        self.max_depth = 3
        self.max_paths = 5
        self.selected_documents = []  # 存储被选中的最有价值文献
        
    def extract_keywords(self, topic):
        """
        使用NLP技术从主题中提取关键词
        
        Args:
            topic: 研究主题
            
        Returns:
            关键词列表
        """
        import re
        
        # 尝试使用jieba进行中文分词和关键词提取
        import jieba
        import jieba.analyse
        
        # # 确保jieba正确初始化
        # if not hasattr(self, '_jieba_initialized'):
        #     # 加载停用词
        #     stop_words = {'的', '是', '在', '有', '和', '与', '或', '但', '而', '了', '着', '过', '等', '及', '以及', '关于', '对于', '进行', '通过', '基于', '研究', '分析'}
        #     for word in stop_words:
        #         jieba.analyse.set_stop_word(word)
            
        #     # 可选：添加自定义词典以提高分词准确性
        #     # jieba.load_userdict('custom_dict.txt')
            
        #     self._jieba_initialized = True
        
        # 1. 使用TF-IDF提取关键词
        tfidf_keywords = jieba.analyse.extract_tags(
            topic, 
            topK=3,  # 提取前3个TF-IDF关键词
            withWeight=False,
            allowPOS=('n', 'vn', 'v', 'a')  # 只提取名词、动词、形容词
        )
        
        # 2. 使用TextRank提取关键词
        textrank_keywords = jieba.analyse.textrank(
            topic, 
            topK=3,  # 提取前3个TextRank关键词
            withWeight=False,
            allowPOS=('n', 'vn', 'v', 'a')
        )
        
        # 3. 合并两种方法的结果，去重并保持顺序
        combined_keywords = []
        seen = set()
        
        # 先添加TF-IDF关键词
        for keyword in tfidf_keywords:
            if keyword not in seen and len(keyword) > 1:
                combined_keywords.append(keyword)
                seen.add(keyword)
        
        # 再添加TextRank关键词（未包含在TF-IDF中的）
        for keyword in textrank_keywords:
            if keyword not in seen and len(keyword) > 1:
                combined_keywords.append(keyword)
                seen.add(keyword)
        
        logger.info(f"提取到的关键词：{combined_keywords}")
        # 如果NLP方法提取到了关键词，返回结果
        if combined_keywords:
            return combined_keywords[:5]  # 最多返回5个关键词
            


    def decide_next_search_terms(self, topic, current_path, search_history):
        """
        使用大模型决定下一步搜索的关键词
        
        Args:
            topic: 研究主题
            current_path: 当前搜索路径
            search_history: 搜索历史记录
            
        Returns:
            下一步搜索的关键词列表
        """
        # 构建当前搜索状态的描述
        current_state = f"""
研究主题：{topic}

当前搜索路径：
"""
        
        for i, step in enumerate(current_path):
            node = step['node']
            current_state += f"步骤{i+1}: {node['name']} ({node.get('node_type', 'Unknown')})\n"
            if node.get('description'):
                current_state += f"  描述: {node['description']}\n"
            if node.get('relations'):
                relations = [r['related_name'] for r in node['relations'][:3]]
                current_state += f"  相关概念: {', '.join(relations)}\n"
        
        current_state += f"""
已搜索过的关键词：{', '.join(search_history)}

已访问的节点数量：{len(self.visited_nodes)}

请分析当前搜索状态，并建议下一步应该搜索的关键词。要求：
1. 基于当前路径中的节点关系和描述，推荐3-5个相关关键词
2. 避免重复已搜索过的关键词
3. 关键词应该有助于扩展研究路径，覆盖主题的不同方面
4. 优先选择能够连接不同研究分支的关键词

请以JSON格式返回关键词列表，例如：
["关键词1", "关键词2", "关键词3"]

你的输出仅包含这个列表，不要输出任何其他提示性语句
"""
        
        try:
            # 调用大模型获取搜索建议
            response = self.llm_client.call_llm(current_state, model_type="chat")
            
            # 尝试解析JSON响应
            import json
            try:
                # 清理响应文本，提取JSON部分
                response_clean = response.strip()
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:]
                if response_clean.endswith('```'):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                keywords = json.loads(response_clean)
                if isinstance(keywords, list):
                    # 过滤掉已搜索过的关键词
                    new_keywords = [kw for kw in keywords if kw not in search_history]
                    return new_keywords[:5]  # 最多返回5个关键词
                else:
                    logger.warning("大模型返回的不是列表格式，使用默认策略")
            except json.JSONDecodeError:
                logger.warning("无法解析大模型返回的JSON，使用默认策略")
            
        except Exception as e:
            logger.error(f"调用大模型决策搜索关键词时出错: {str(e)}")
        
        # 如果大模型调用失败，使用默认策略
        return self._fallback_search_strategy(current_path, search_history)
    
    def _fallback_search_strategy(self, current_path, search_history):
        """
        备用搜索策略：基于当前路径中的关系生成关键词
        
        Args:
            current_path: 当前搜索路径
            search_history: 搜索历史记录
            
        Returns:
            关键词列表
        """
        keywords = []
        
        # 从当前路径的节点关系中提取关键词
        for step in current_path:
            node = step['node']
            if node.get('relations'):
                for relation in node['relations']:
                    related_name = relation['related_name']
                    if related_name not in search_history and related_name not in keywords:
                        keywords.append(related_name)
                        if len(keywords) >= 3:
                            break
                if len(keywords) >= 3:
                    break
        
        # 如果没有足够的关键词，从节点名称中提取
        if len(keywords) < 2:
            for step in current_path[-2:]:  # 只考虑最后两个节点
                node = step['node']
                if node['name'] not in search_history and node['name'] not in keywords:
                    keywords.append(node['name'])
                    if len(keywords) >= 3:
                        break
        
        return keywords[:3]
    
    def recommend_documents_for_node(self, node, topic, max_docs=20):
        """
        为特定节点推荐重要的文献
        
        Args:
            node: 节点信息
            topic: 研究主题
            max_docs: 最大推荐文献数量
            
        Returns:
            推荐的文献列表
        """
        if not node.get('source_documents'):
            return []
        
        # 获取节点的所有文献
        documents = []
        for doc in node['source_documents']:
            if isinstance(doc, str):
                try:
                    import json
                    doc = json.loads(doc)
                except json.JSONDecodeError:
                    continue
            
            if isinstance(doc, dict):
                documents.append(doc)
        
        if not documents:
            return []
        
        # 如果文献数量少于等于最大推荐数量，直接返回
        if len(documents) <= max_docs:
            return documents
        
        # 使用大模型智能推荐文献
        try:
            # 构建文献推荐提示
            prompt = f"""
请根据研究主题从以下文献列表中推荐重要的{max_docs}篇文献：

研究主题：{topic}
节点名称：{node['name']}
节点类型：{node.get('node_type', 'Unknown')}

文献列表：
"""
            
            for i, doc in enumerate(documents, 1):
                title = doc.get('title', '未知标题')
                filename = doc.get('filename', '未知文件')
                prompt += f"{i}. 标题: {title}\n   文件: {filename}\n"
            
            prompt += f"""
请基于与研究主题的相关性、文献质量、重要性等因素，推荐重要的{max_docs}篇文献。
请以JSON格式返回推荐文献的索引列表，例如：
[1, 3, 5]

推荐文献索引："""
            
            # 调用大模型获取推荐
            response = self.llm_client.call_llm(prompt, model_type="chat")
            
            # 解析推荐结果
            import json
            try:
                # 清理响应文本
                response_clean = response.strip()
                if response_clean.startswith('```json'):
                    response_clean = response_clean[7:]
                if response_clean.endswith('```'):
                    response_clean = response_clean[:-3]
                response_clean = response_clean.strip()
                
                recommended_indices = json.loads(response_clean)
                if isinstance(recommended_indices, list):
                    # 根据推荐索引获取文献
                    recommended_docs = []
                    for idx in recommended_indices:
                        if 1 <= idx <= len(documents):
                            recommended_docs.append(documents[idx - 1])
                        if len(recommended_docs) >= max_docs:
                            break
                    
                    if recommended_docs:
                        logger.info(f"为节点 '{node['name']}' 推荐了 {len(recommended_docs)} 篇文献")
                        return recommended_docs
            
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"解析文献推荐结果失败: {str(e)}")
        
        except Exception as e:
            logger.warning(f"文献推荐过程出错: {str(e)}")
        
        # 如果智能推荐失败，使用简单的策略：选择前max_docs篇
        logger.info(f"使用默认策略为节点 '{node['name']}' 选择前 {max_docs} 篇文献")
        return documents[:max_docs]

    def find_multiple_paths(self, topic):
        """
        查找多条能够解决问题的路径
        增强版：使用大模型智能决策搜索方向
        
        Args:
            topic: 研究主题
            
        Returns:
            多条路径的列表
        """
        # 记录Agent开始工作
        if self.report_generator:
            self.report_generator.record_agent_start("search_agent")
        
        logger.info(f"开始为主题 '{topic}' 查找多条路径（使用大模型智能搜索）...")
        
        try:
            # 重置状态
            self.visited_nodes.clear()
            self.search_paths.clear()
            
            # 搜索历史记录
            search_history = set()
            
            # 初始搜索：使用主题关键词
            initial_keywords = self.extract_keywords(topic)
            search_history.update(initial_keywords)
            
            # 执行初始搜索
            all_initial_results = []
            for keyword in initial_keywords:
                try:
                    results = self.kg.search_knowledge_by_text(
                        keyword, 
                        top_k=3, 
                        max_hops=2, 
                        dot_threshold=0.95
                    )
                    
                    for result in results:
                        node_id = result['id']
                        if node_id not in self.visited_nodes:
                            self.visited_nodes.add(node_id)
                            
                            # 解码source_documents
                            if 'source_documents' in result and result['source_documents']:
                                result['source_documents'] = decode_unicode_in_source_documents(result['source_documents'])
                            
                            all_initial_results.append({
                                'path': [{
                                    'node': result,
                                    'keyword': keyword,
                                    'depth': 0
                                }],
                                'current_node': result,
                                'keyword': keyword
                            })
                    
                except Exception as e:
                    logger.warning(f"搜索关键词 '{keyword}' 时出错: {str(e)}")
                    continue
            
            # 为每条初始路径使用大模型智能扩展
            for result in all_initial_results:
                if len(self.search_paths) >= self.max_paths:
                    break
                    
                current_path = result['path']
                current_node = result['current_node']
                
                # 使用大模型决定下一步搜索方向
                next_keywords = self.decide_next_search_terms(topic, current_path, list(search_history))
                
                if not next_keywords:
                    # 如果大模型没有返回关键词，使用备用策略
                    next_keywords = self._fallback_search_strategy(current_path, list(search_history))
                
                # 执行智能扩展搜索
                for keyword in next_keywords[:2]:  # 限制扩展搜索的关键词数量
                    if len(self.search_paths) >= self.max_paths:
                        break
                        
                    search_history.add(keyword)
                    
                    try:
                        extended_results = self.kg.search_knowledge_by_text(
                            keyword, 
                            top_k=2, 
                            max_hops=2, 
                            dot_threshold=0.95
                        )
                        
                        for extended_result in extended_results:
                            if len(self.search_paths) >= self.max_paths:
                                break
                                
                            extended_node_id = extended_result['id']
                            if extended_node_id not in self.visited_nodes:
                                self.visited_nodes.add(extended_node_id)
                                
                                # 解码source_documents
                                if 'source_documents' in extended_result and extended_result['source_documents']:
                                    extended_result['source_documents'] = decode_unicode_in_source_documents(extended_result['source_documents'])
                                
                                # 构建扩展路径
                                extended_path = current_path + [{
                                    'node': extended_result,
                                    'keyword': keyword,
                                    'depth': 1
                                }]
                                
                                self.search_paths.append(extended_path)
                                
                                # 进一步扩展相似节点
                                if extended_result.get('similar_nodes') and len(current_path) < self.max_depth - 1:
                                    for similar_node in extended_result['similar_nodes'][:1]:  # 限制相似节点数量
                                        if similar_node['id'] not in self.visited_nodes:
                                            similar_results = self.kg.get_node_relations(
                                                similar_node['id'], 
                                                similar_node.get('type', 'Unknown'), 
                                                limit=2
                                            )
                                            
                                            similar_node_full = {
                                                'id': similar_node['id'],
                                                'name': similar_node['name'],
                                                'node_type': similar_node.get('type', 'Unknown'),
                                                'relations': similar_results,
                                                'similarity': 0.8,
                                                'source_documents': []
                                            }
                                            
                                            self.visited_nodes.add(similar_node['id'])
                                            
                                            final_path = extended_path + [{
                                                'node': similar_node_full,
                                                'keyword': f"similar_to_{keyword}",
                                                'depth': 2
                                            }]
                                            
                                            if len(self.search_paths) < self.max_paths:
                                                self.search_paths.append(final_path)
                    
                    except Exception as e:
                        logger.warning(f"扩展搜索关键词 '{keyword}' 时出错: {str(e)}")
                        continue
            
            # 如果没有找到足够的路径，添加一些初始结果
            if len(self.search_paths) < 2:
                for result in all_initial_results:
                    if len(self.search_paths) >= self.max_paths:
                        break
                    self.search_paths.append(result['path'])
            
            logger.info(f"找到 {len(self.search_paths)} 条路径（使用大模型智能搜索）")
            
            # 记录Agent完成工作
            if self.report_generator:
                search_output_data = {
                    "keywords": initial_keywords,
                    "search_paths": self.search_paths,
                    "total_paths": len(self.search_paths),
                    "total_nodes": len(self.visited_nodes),
                    "raw_output": f"搜索完成，找到 {len(self.search_paths)} 条路径，包含 {len(self.visited_nodes)} 个节点"
                }
                self.report_generator.record_agent_end("search_agent", search_output_data)
            
            return self.search_paths
            
        except Exception as e:
            error_msg = f"搜索过程中出错: {str(e)}"
            logger.error(error_msg)
            if self.report_generator:
                self.report_generator.record_error(error_msg, "search_agent")
            return []
    
    def _get_path_relation_documents(self, path):
        """
        从路径中提取关系文献信息
        
        Args:
            path: 当前搜索路径
            
        Returns:
            路径中关系的文献列表
        """
        relation_documents = []
        
        # 遍历路径中的每个步骤，检查是否包含关系信息
        for i, step in enumerate(path):
            # 检查当前步骤是否包含关系信息
            if 'relations' in step:
                for relation in step['relations']:
                    # 如果关系中包含source_documents
                    if 'source_documents' in relation:
                        for doc in relation['source_documents']:
                            if isinstance(doc, str):
                                try:
                                    import json
                                    doc = json.loads(doc)
                                except json.JSONDecodeError:
                                    continue
                            
                            if isinstance(doc, dict):
                                # 添加关系信息到文献中
                                doc_with_relation = doc.copy()
                                doc_with_relation['relation_type'] = relation.get('relation_type', 'UNKNOWN')
                                doc_with_relation['relation_source'] = relation.get('source', 'UNKNOWN')
                                doc_with_relation['relation_target'] = relation.get('target', 'UNKNOWN')
                                relation_documents.append(doc_with_relation)
            
            # 检查是否有next_relation字段
            if 'next_relation' in step:
                next_relation = step['next_relation']
                if 'source_documents' in next_relation:
                    for doc in next_relation['source_documents']:
                        if isinstance(doc, str):
                            try:
                                import json
                                doc = json.loads(doc)
                            except json.JSONDecodeError:
                                continue
                        
                        if isinstance(doc, dict):
                            doc_with_relation = doc.copy()
                            doc_with_relation['relation_type'] = next_relation.get('type', 'UNKNOWN')
                            doc_with_relation['relation_source'] = next_relation.get('source', 'UNKNOWN')
                            doc_with_relation['relation_target'] = next_relation.get('target', 'UNKNOWN')
                            relation_documents.append(doc_with_relation)
        
        return relation_documents
    
    def _get_node_documents(self, node_name, path, topic=None):
        """
        获取特定节点的文献信息和路径上关系中的文献信息，使用智能推荐
        
        Args:
            node_name: 节点名称
            path: 当前路径
            topic: 研究主题（用于智能推荐）
            
        Returns:
            该节点及其路径上关系相关的文献列表
        """
        all_documents = []
        node_info = None
        
        # 遍历路径中的每个步骤，找到对应节点的文献
        for step in path:
            node = step['node']
            if node['name'] == node_name:
                node_info = node
                break
        
        if not node_info:
            return all_documents
        
        # 收集节点的文献
        node_documents = []
        if node_info.get('source_documents'):
            for doc in node_info['source_documents']:
                if isinstance(doc, str):
                    try:
                        import json
                        doc = json.loads(doc)
                    except json.JSONDecodeError:
                        logger.warning(f"无法解析文档JSON: {doc}")
                        continue
                
                if isinstance(doc, dict):
                    node_documents.append(doc)
        
        # 从路径中收集关系的文献
        relation_documents = self._get_path_relation_documents(path)
        logger.info(f"节点 '{node_name}' 从路径关系中获取了 {len(relation_documents)} 篇文献")
        
        # 合并节点和关系的文献
        merged_documents = node_documents + relation_documents
        
        # 如果提供了主题，使用智能推荐
        if topic and self.llm_client and merged_documents:
            # 使用合并后的文献进行推荐
            recommended_docs = self.recommend_documents_for_node({
                'name': node_name,
                'node_type': node_info.get('node_type', 'Concept'),
                'source_documents': merged_documents
            }, topic, max_docs=20)
            
            for doc in recommended_docs:
                all_documents.append({
                    'filename': doc.get('filename', '未知文件'),
                    'title': doc.get('title', '未知标题'),
                    'paragraphs': [doc.get('paragraph_number')] if doc.get('paragraph_number') else [],
                    'source': 'relation' if doc.get('relation_type') else 'node'
                })
            logger.info(f"节点 '{node_name}' 使用智能推荐，从节点和路径关系文献中选择了 {len(recommended_docs)} 篇文献")
        else:
            # 直接使用所有文献
            for doc in merged_documents:
                all_documents.append({
                    'filename': doc.get('filename', '未知文件'),
                    'title': doc.get('title', '未知标题'),
                    'paragraphs': [doc.get('paragraph_number')] if doc.get('paragraph_number') else [],
                    'source': 'relation' if doc.get('relation_type') else 'node'
                })
        
        return all_documents
    
    def get_top_k_documents(self, all_documents, topic, k=10):
        """
        从所有文献中选择最有价值的k篇
        
        Args:
            all_documents: 所有收集到的文献列表
            topic: 研究主题
            k: 需要返回的文献数量
            
        Returns:
            最有价值的k篇文献列表
        """
        if not all_documents:
            return []
        
        # 如果文献数量少于等于k，直接返回
        if len(all_documents) <= k:
            return all_documents
        
        # 使用大模型进行文献价值排序
        if self.llm_client:
            try:
                # 构建文献排序提示
                prompt = f"""
                请根据以下文献与研究主题"{topic}"的相关性和价值，为这些文献排序，并返回最有价值的前{k}篇文献的索引（从0开始）。
                文献所属的节点信息和关系信息对评估文献价值非常重要，请在评估时充分考虑文献与知识图谱结构的关联性。
                请只返回索引列表，不要返回任何其他内容。
                
                文献列表：
                """
                
                for i, doc in enumerate(all_documents):
                    # 根据文献来源类型生成不同的描述信息
                    source_type = doc.get('source_type', 'unknown')
                    if source_type == 'node':
                        node_name = doc.get('node_name', '未知节点')
                        doc_info = f"{i}. 标题: {doc.get('title', '未知标题')}, 文件名: {doc.get('filename', '未知文件')}, 来源: 节点({node_name})"
                    elif source_type == 'relation':
                        relation_type = doc.get('relation_type', 'UNKNOWN')
                        source_node = doc.get('relation_source', 'UNKNOWN')
                        target_node = doc.get('relation_target', 'UNKNOWN')
                        doc_info = f"{i}. 标题: {doc.get('title', '未知标题')}, 文件名: {doc.get('filename', '未知文件')}, 来源: 关系({source_node}->{relation_type}->{target_node})"
                    else:
                        doc_info = f"{i}. 标题: {doc.get('title', '未知标题')}, 文件名: {doc.get('filename', '未知文件')}, 来源: 未知"
                    prompt += doc_info + "\n"
                
                # 调用大模型排序
                response = self.llm_client.generate_content(
                    prompt,
                    temperature=0.0
                )
                
                # 解析返回的索引列表
                try:
                    # 尝试直接解析JSON
                    import json
                    top_indices = json.loads(response)
                except (json.JSONDecodeError, TypeError):
                    # 如果不是有效的JSON，尝试解析文本中的数字列表
                    import re
                    indices = re.findall(r'\d+', response)
                    top_indices = [int(idx) for idx in indices[:k]]
                
                # 根据索引获取对应的文献
                top_documents = []
                for idx in top_indices:
                    if 0 <= idx < len(all_documents):
                        top_documents.append(all_documents[idx])
                

                
                logger.info(f"成功从{len(all_documents)}篇文献中选择了{len(top_documents)}篇最有价值的文献")
                return top_documents
                
            except Exception as e:
                logger.error(f"文献排序过程中出错: {str(e)}")
        
        # 降级策略：优先选择包含摘要的文献
        logger.warning("使用降级策略选择文献")
        documents_with_summary = [doc for doc in all_documents if doc.get('summary')]
        documents_without_summary = [doc for doc in all_documents if not doc.get('summary')]
        
        top_documents = documents_with_summary[:min(k, len(documents_with_summary))]
        remaining = k - len(top_documents)
        if remaining > 0:
            top_documents.extend(documents_without_summary[:remaining])
        
        return top_documents
    
    def select_top_documents_from_paths(self, search_paths, topic, max_documents=10):
        """
        从搜索路径中提取所有文献（包括节点和关系中的文献），并选择最有价值的k篇文献
        
        Args:
            search_paths: 搜索路径列表
            topic: 研究主题
            max_documents: 最多返回的文献数量
            
        Returns:
            包含所有文档的列表和被选中的最有价值文档列表
        """
        all_documents = []
        document_filenames = set()  # 用于去重
        node_doc_count = 0  # 统计节点文献数量
        relation_doc_count = 0  # 统计关系文献数量
        
        # 从所有路径中提取文档
        for path in search_paths:
            for step in path:
                # 处理节点中的文献
                node = step.get('node', {})
                if node.get('source_documents'):
                    for doc in node['source_documents']:
                        # 处理文档格式
                        if isinstance(doc, str):
                            try:
                                import json
                                doc = json.loads(doc)
                            except json.JSONDecodeError:
                                continue
                        
                        if isinstance(doc, dict):
                            filename = doc.get('filename') or doc.get('name')
                            if filename and filename not in document_filenames:
                                document_filenames.add(filename)
                                node_doc_count += 1
                                all_documents.append({
                                    'filename': filename,
                                    'title': doc.get('title', filename),
                                    'node_name': node.get('name', 'Unknown'),
                                    'raw_doc': doc,
                                    'source_type': 'node'
                                })
                
                # 处理关系中的文献
                # 处理relations字段中的关系
                if 'relations' in step:
                    for relation in step['relations']:
                        if 'source_documents' in relation:
                            for doc in relation['source_documents']:
                                if isinstance(doc, str):
                                    try:
                                        import json
                                        doc = json.loads(doc)
                                    except json.JSONDecodeError:
                                        continue
                                
                                if isinstance(doc, dict):
                                    filename = doc.get('filename') or doc.get('name')
                                    if filename and filename not in document_filenames:
                                        document_filenames.add(filename)
                                        relation_doc_count += 1
                                        all_documents.append({
                                            'filename': filename,
                                            'title': doc.get('title', filename),
                                            'relation_type': relation.get('relation_type', 'UNKNOWN'),
                                            'relation_source': relation.get('source', 'UNKNOWN'),
                                            'relation_target': relation.get('target', 'UNKNOWN'),
                                            'raw_doc': doc,
                                            'source_type': 'relation'
                                        })
                
                # 处理next_relation字段中的关系
                if 'next_relation' in step:
                    next_relation = step['next_relation']
                    if 'source_documents' in next_relation:
                        for doc in next_relation['source_documents']:
                            if isinstance(doc, str):
                                try:
                                    import json
                                    doc = json.loads(doc)
                                except json.JSONDecodeError:
                                    continue
                            
                            if isinstance(doc, dict):
                                filename = doc.get('filename') or doc.get('name')
                                if filename and filename not in document_filenames:
                                    document_filenames.add(filename)
                                    relation_doc_count += 1
                                    all_documents.append({
                                        'filename': filename,
                                        'title': doc.get('title', filename),
                                        'relation_type': next_relation.get('type', 'UNKNOWN'),
                                        'relation_source': next_relation.get('source', 'UNKNOWN'),
                                        'relation_target': next_relation.get('target', 'UNKNOWN'),
                                        'raw_doc': doc,
                                        'source_type': 'relation'
                                    })
        
        logger.info(f"从搜索路径中提取了{len(all_documents)}篇唯一文献（节点文献: {node_doc_count}篇，关系文献: {relation_doc_count}篇）")
        
        # 选择最有价值的k篇文献
        if all_documents and topic:
            self.selected_documents = self.get_top_k_documents(all_documents, topic, max_documents)
            logger.info(f"已选择{len(self.selected_documents)}篇最有价值的文献进行深入处理")
        else:
            self.selected_documents = all_documents[:max_documents] if max_documents > 0 else all_documents
        
        return all_documents, self.selected_documents
    
    def format_search_results(self, topic=None, max_documents=10):
        """
        格式化搜索结果为文献综述所需的上下文
        支持显示基于主题的文档概括信息
        
        Args:
            topic: 研究主题，用于智能推荐文献
            max_documents: 最多返回的文献数量
            
        Returns:
            格式化的上下文字符串
        """
        if not self.search_paths:
            return "未找到相关的研究路径。"
        
        context_parts = []
        
        for i, path in enumerate(self.search_paths):
            context_parts.append(f"\n=== 研究路径 {i+1} ===")
            
            # 收集所有节点和文档信息
            all_nodes = []
            all_document_summaries = []  # 收集所有基于主题的文档概括
            all_documents = []  # 收集所有文献用于后续排序
            
            for step in path:
                node = step['node']
                keyword = step['keyword']
                depth = step['depth']
                
                all_nodes.append({
                    'name': node['name'],
                    'type': node.get('node_type', node.get('type', 'Unknown')),
                    'keyword': keyword,
                    'depth': depth,
                    'description': node.get('description', ''),
                    'relations': node.get('relations', []),
                    'document_summaries': node.get('document_summaries', []),
                    'node_summary': node.get('node_summary', '')
                })
                
                # 收集基于主题的文档概括
                if node.get('document_summaries'):
                    for doc_sum in node['document_summaries']:
                        doc_info = {
                            'node_name': node['name'],
                            'filename': doc_sum['filename'],
                            'title': doc_sum['title'],
                            'summary': doc_sum['topic_aware_summary'],
                            'citation': doc_sum.get('citation', '')
                        }
                        all_document_summaries.append(doc_info)
                        # 同时收集文献信息用于排序
                        all_documents.append({
                            'filename': doc_sum['filename'],
                            'title': doc_sum['title'],
                            'summary': doc_sum['topic_aware_summary']
                        })
            
            # 构建路径描述
            path_description = f"路径深度: {len(path)} 步\n"
            path_description += "核心概念: " + " -> ".join([node['name'] for node in all_nodes]) + "\n"
            
            # 添加节点详细信息
            path_description += "\n关键节点信息:\n"
            for node in all_nodes:
                path_description += f"- {node['name']} ({node['type']})\n"
                if node['description']:
                    path_description += f"  📝 描述: {node['description']}\n"
                if node['relations']:
                    path_description += f"  🔗 相关关系: {', '.join([r['related_name'] for r in node['relations']])}\n"
                
                # 添加节点级别的概括
                if node['node_summary']:
                    path_description += f"  📋 节点概括: {node['node_summary']}\n"
                
                # 添加相关文献信息
                if node['document_summaries']:
                    path_description += f"  📚 相关文献:\n"
                    for doc in node['document_summaries']:
                        path_description += f"    • 《{doc['title']}》 ({doc['filename']})\n"
            
            # 在路径末尾添加基于主题的文档深度概括信息
            if all_document_summaries:
                # 获取最有价值的文档概括
                if topic and max_documents > 0 and len(all_document_summaries) > max_documents:
                    # 先根据文献信息获取最有价值的文献
                    top_documents = self.get_top_k_documents(all_documents, topic, max_documents)
                    # 创建文献标题到文档概括的映射
                    doc_title_to_summary = {doc['title']: doc for doc in all_document_summaries}
                    # 根据最有价值的文献筛选文档概括
                    top_document_summaries = []
                    for top_doc in top_documents:
                        if top_doc['title'] in doc_title_to_summary:
                            top_document_summaries.append(doc_title_to_summary[top_doc['title']])
                    # 如果筛选结果不足，使用原始文档概括
                    if not top_document_summaries:
                        top_document_summaries = all_document_summaries[:max_documents]
                    logger.info(f"从{len(all_document_summaries)}篇文档概括中选择了{len(top_document_summaries)}篇最有价值的文档概括")
                else:
                    # 如果不需要排序或文献数量不足，直接使用所有文献概括
                    top_document_summaries = all_document_summaries[:max_documents] if max_documents > 0 else all_document_summaries
                
                path_description += "\n📖 基于主题的文档深度概括:\n"
                path_description += "-" * 40 + "\n"
                
                for i, doc_summary in enumerate(top_document_summaries, 1):
                    path_description += f"\n{i}. 《{doc_summary['title']}》\n"
                    path_description += f"   📁 文件: {doc_summary['filename']}\n"
                    path_description += f"   🔍 所属节点: {doc_summary['node_name']}\n"
                    if doc_summary.get('citation'):
                        path_description += f"   📖 引用: {doc_summary['citation']}\n"
                    path_description += f"   📝 主题相关概括:\n"
                    path_description += f"   {doc_summary['summary']}\n"
                
                path_description += "\n" + "-" * 40 + "\n"
            
            context_parts.append(path_description)
        
        return "\n".join(context_parts)

class DocumentReaderAgent:
    """
    文档阅读Agent：负责使用大模型根据研究主题概括文献内容
    """
    
    def __init__(self, document_manager=None, llm_client=None, report_generator=None):
        """
        初始化文档阅读Agent
        
        Args:
            document_manager: DocumentManager实例
            llm_client: LLMClient实例
            report_generator: ReportGenerator实例
        """
        self.document_manager = document_manager or DocumentManager()
        self.llm_client = llm_client
        self.report_generator = report_generator
        self.logger = logging.getLogger(__name__)
    
    def process_documents_from_paths(self, search_paths, topic=None, selected_documents=None, max_workers=4):
        """
        处理搜索路径中的文档，使用大模型根据主题概括文献
        每篇文献单独调用大模型，防止token溢出
        如果提供了selected_documents参数，则只处理指定的文献
        
        Args:
            search_paths: 搜索Agent找到的路径列表
            topic: 研究主题，用于生成有针对性的文档概括
            selected_documents: 要处理的文献文件名列表（可选）
            max_workers: 最大工作线程数，默认4
            
        Returns:
            处理后的路径，包含基于主题的文档概括信息
        """
        # 记录Agent开始工作
        if self.report_generator:
            self.report_generator.record_agent_start("document_reader_agent")
        
        try:
            self.logger.info(f"开始处理搜索路径中的文档...（主题: {topic}，并行工作线程数: {max_workers}）")
            
            # 首先收集所有需要处理的文档并进行去重
            all_documents = []
            processed_filenames = set()  # 用于去重的文件名集合
            
            for path in search_paths:
                for step in path:
                    node = step['node']
                    
                    if node.get('source_documents'):
                        for doc in node['source_documents']:
                            # 处理文档信息格式
                            if isinstance(doc, str):
                                try:
                                    import json
                                    doc = json.loads(doc)
                                except json.JSONDecodeError:
                                    self.logger.warning(f"无法解析文档JSON: {doc}")
                                    continue
                            
                            if isinstance(doc, dict):
                                filename = doc.get('filename', 'Unknown')
                                title = doc.get('title', 'Unknown title')
                                
                                # 去重：如果文件名已经处理过，跳过
                                if filename not in processed_filenames:
                                    processed_filenames.add(filename)
                                    all_documents.append({
                                        'filename': filename,
                                        'title': title,
                                        'doc_info': doc
                                    })
                                else:
                                    self.logger.info(f"跳过重复文档: {filename}")
            
            # 如果提供了selected_documents参数，只保留选中的文档
            if selected_documents:
                original_count = len(all_documents)
                all_documents = [doc for doc in all_documents if doc['filename'] in selected_documents]
                self.logger.info(f"从{original_count}个文档中筛选出{len(all_documents)}个被选中的文档进行处理")
            
            self.logger.info(f"准备并行处理 {len(all_documents)} 个文档...")
            
            # 批量处理所有唯一文档（并行方式）
            document_summaries_cache = {}
            active_processes = 0
            
            # 定义处理单个文档的函数
            def process_single_document(doc_info):
                filename = doc_info['filename']
                title = doc_info['title']
                try:
                    self.logger.info(f"开始处理文档: {title}")
                    # 调用大模型进行概括
                    summary = self._summarize_document_with_topic(filename, title, topic)
                    
                    result = {
                        'filename': filename,
                        'title': title,
                        'topic_aware_summary': summary,
                        'citation': self._generate_citation(doc_info['doc_info'])
                    }
                    self.logger.info(f"文档 {title} 处理完成")
                    return result
                except Exception as e:
                    self.logger.error(f"处理文档 {title} 时出错: {str(e)}")
                    # 返回错误信息，以便后续处理
                    return {
                        'filename': filename,
                        'title': title,
                        'topic_aware_summary': f"处理失败: {str(e)}",
                        'citation': self._generate_citation(doc_info['doc_info']),
                        'error': str(e)
                    }
            
            # 使用线程池并行处理文档
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(max_workers, len(all_documents))) as executor:
                # 提交所有文档处理任务
                future_to_doc = {
                    executor.submit(process_single_document, doc_info): doc_info 
                    for doc_info in all_documents
                }
                
                # 收集处理结果
                for future in concurrent.futures.as_completed(future_to_doc):
                    doc_info = future_to_doc[future]
                    try:
                        result = future.result()
                        document_summaries_cache[result['filename']] = result
                    except Exception as e:
                        self.logger.error(f"获取文档 {doc_info['title']} 结果时出错: {str(e)}")
                        # 即使出错也保存基本信息，避免后续处理失败
                        document_summaries_cache[doc_info['filename']] = {
                            'filename': doc_info['filename'],
                            'title': doc_info['title'],
                            'topic_aware_summary': f"处理结果获取失败: {str(e)}",
                            'citation': self._generate_citation(doc_info['doc_info']),
                            'error': str(e)
                        }
            
            # 现在重新构建路径，使用缓存的文档概括结果
            processed_paths = []
            
            for i, path in enumerate(search_paths):
                self.logger.info(f"构建处理后的路径 {i+1}/{len(search_paths)}")
                
                processed_path = []
                
                for step in path:
                    node = step['node']
                    
                    # 处理该节点的文档
                    if node.get('source_documents'):
                        node_document_summaries = []
                        
                        for doc in node['source_documents']:
                            # 处理文档信息格式
                            if isinstance(doc, str):
                                try:
                                    import json
                                    doc = json.loads(doc)
                                except json.JSONDecodeError:
                                    self.logger.warning(f"无法解析文档JSON: {doc}")
                                    continue
                            
                            if isinstance(doc, dict):
                                filename = doc.get('filename', 'Unknown')
                                
                                # 从缓存中获取文档概括
                                if filename in document_summaries_cache:
                                    node_document_summaries.append(document_summaries_cache[filename])
                                else:
                                    self.logger.warning(f"未找到文档 {filename} 的处理结果")
                        
                        # 更新节点信息，添加基于主题的文档概括
                        updated_node = node.copy()
                        updated_node['document_summaries'] = node_document_summaries
                        
                        # 创建节点级别的概括
                        if node_document_summaries:
                            node_summary = self._create_node_summary(node_document_summaries, topic)
                            updated_node['node_summary'] = node_summary
                        
                        processed_step = step.copy()
                        processed_step['node'] = updated_node
                        processed_path.append(processed_step)
                    else:
                        processed_path.append(step)
                
                processed_paths.append(processed_path)
            
            self.logger.info(f"文档处理完成，共处理 {len(all_documents)} 个唯一文档")
            
            # 记录Agent成功完成工作
            if self.report_generator:
                doc_output_data = {
                    'topic': topic,
                    'total_documents': len(all_documents),
                    'selected_documents_count': len(selected_documents) if selected_documents else "全部",
                    'processed_paths': len(processed_paths),
                    'document_summaries': list(document_summaries_cache.values())
                }
                self.report_generator.record_agent_end("document_reader_agent", doc_output_data)
            
            return processed_paths
            
        except Exception as e:
            # 记录错误
            if self.report_generator:
                self.report_generator.record_error("document_reader_agent", str(e))
            self.logger.error(f"文档处理失败: {str(e)}")
            raise
    
    def _find_document_path(self, filename):
        """
        查找文档文件的完整路径
        
        Args:
            filename: 文件名
            
        Returns:
            文档的完整路径，如果未找到则返回None
        """
        # 常见的文档目录
        search_dirs = [
            '.',  # 当前目录
            'documents',  # documents目录
            'papers',  # papers目录
            'literature',  # literature目录
            'pdfs',  # pdfs目录
        ]
        
        # 在当前目录及其子目录中搜索
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for file in files:
                        if file == filename:
                            return os.path.join(root, file)
        
        # 如果没找到，尝试直接使用文件名
        if os.path.exists(filename):
            return filename
        
        return None
    
    def _generate_citation(self, doc_info):
        """
        生成文献的标准学术引用格式
        
        Args:
            doc_info: 文档信息字典
            
        Returns:
            标准格式的引用字符串
        """
        try:
            # 获取文献信息
            title = doc_info.get('title', '未知标题')
            filename = doc_info.get('filename', '未知文件')
            authors = doc_info.get('authors', [])
            year = doc_info.get('year', '')
            journal = doc_info.get('journal', '')
            volume = doc_info.get('volume', '')
            pages = doc_info.get('pages', '')
            doi = doc_info.get('doi', '')
            
            # 生成作者字符串
            if authors:
                if isinstance(authors, list):
                    if len(authors) <= 3:
                        author_str = ', '.join(authors)
                    else:
                        author_str = ', '.join(authors[:3]) + ', et al.'
                else:
                    author_str = str(authors)
            else:
                # 如果没有作者信息，从文件名提取
                author_str = self._extract_author_from_filename(filename)
            
            # 构建引用格式
            citation_parts = []
            
            if author_str:
                citation_parts.append(author_str)
            
            if year:
                citation_parts.append(f"({year})")
            
            if title:
                citation_parts.append(f"{title}")
            
            if journal:
                journal_part = journal
                if volume:
                    journal_part += f", {volume}"
                if pages:
                    journal_part += f", {pages}"
                citation_parts.append(f"*{journal_part}*")
            
            if doi:
                citation_parts.append(f"DOI: {doi}")
            
            # 如果没有足够的信息，使用文件名作为后备
            if len(citation_parts) < 2:
                citation_parts = [f"[{filename}]", title if title else "未知标题"]
            
            citation = '. '.join(citation_parts)
            
            # 确保引用以句号结尾
            if not citation.endswith('.'):
                citation += '.'
            
            return citation
            
        except Exception as e:
            self.logger.warning(f"生成引用时出错: {str(e)}")
            # 返回基本的引用格式
            title = doc_info.get('title', '未知标题')
            filename = doc_info.get('filename', '未知文件')
            return f"[{filename}] {title}."
    
    def _extract_author_from_filename(self, filename):
        """
        从文件名提取作者信息
        
        Args:
            filename: 文件名
            
        Returns:
            推测的作者信息
        """
        try:
            # 移除文件扩展名
            name_without_ext = os.path.splitext(filename)[0]
            
            # 尝试匹配常见的学术命名模式
            import re
            
            # 匹配 "Author_Year_Title" 格式
            match = re.match(r'^([A-Za-z]+)_(\d{4})_', name_without_ext)
            if match:
                return match.group(1)
            
            # 匹配 "Author et al_Year" 格式
            match = re.match(r'^([A-Za-z]+(?:\s+et\s+al)?)_(\d{4})', name_without_ext)
            if match:
                return match.group(1)
            
            # 如果无法匹配，返回文件名的前部分
            parts = name_without_ext.split('_')[:2]
            return '_'.join(parts)
            
        except Exception:
            return "Unknown"
    
    def _summarize_document_with_topic(self, filename, title, topic):
        """
        使用大模型根据研究主题概括单个文档内容
        每篇文献单独调用大模型，防止token溢出
        
        Args:
            filename: 文件名
            title: 文档标题
            topic: 研究主题
            
        Returns:
            基于主题的文档概括
        """
        try:
            # 读取文档内容
            document_text = self.document_manager.read_document(filename)
            
            if not document_text:
                return f"无法读取文档内容: {filename}"
            
            # 更严格的内容长度限制，确保每篇文档单独处理时不会token溢出
            max_content_length = 8000  # 减少内容长度限制
            if len(document_text) > max_content_length:
                document_text = document_text[:max_content_length] + "...[内容已截断]"
            
            # 构建更简洁的主题感知概括提示
            prompt = f"""
请根据研究主题对以下文档进行概括：

研究主题：{topic}
文档标题：{title}

文档内容：
{document_text}

请提供简洁的学术概括（200-300字），重点突出：
1. 与研究主题的相关性
2. 核心观点和贡献
3. 理论或实践价值
4. 重要概念和方法

概括："""
            
            # 调用大模型生成概括（每篇文档单独调用）
            response = self.llm_client.call_llm(prompt, model_type="chat")
            
            # 清理和验证响应
            summary = response.strip()
            if not summary:
                return "大模型未能生成有效概括"
            
            # 确保概括长度合理
            if len(summary) > 1000:
                summary = summary[:1000] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"概括文档 {filename} 时出错: {str(e)}")
            return f"概括文档时出错: {str(e)}"
    
    def _create_node_summary(self, document_summaries, topic):
        """
        为节点创建综合概括
        
        Args:
            document_summaries: 文档概括列表
            topic: 研究主题
            
        Returns:
            节点级别的综合概括
        """
        try:
            # 构建节点概括提示
            prompt = f"""
请基于以下多个文档的概括，为研究主题创建一个综合性的节点概括：

研究主题：{topic}

相关文档概括：
"""
            
            for i, doc_sum in enumerate(document_summaries, 1):
                prompt += f"""
文档 {i}: {doc_sum['title']}
概括: {doc_sum['topic_aware_summary']}
"""
            
            prompt += """

请创建一个综合概括，包括：
1. 这些文档共同为研究主题提供了什么视角
2. 该节点在研究主题中的核心作用
3. 主要的理论框架和实践方法
4. 与其他研究方向的关联性

请以简洁、学术的语调进行概括，字数控制在200-300字之间。
"""
            
            # 调用大模型生成节点概括
            response = self.llm_client.call_llm(prompt, model_type="chat")
            
            # 清理响应
            summary = response.strip()
            if len(summary) > 5000:
                summary = summary[:5000] + "..."
            
            return summary
            
        except Exception as e:
            self.logger.error(f"创建节点概括时出错: {str(e)}")
            return f"创建节点概括时出错: {str(e)}"
    
    def _create_document_summary_text(self, successful_docs, topic=None):
        """
        创建文档概括文本
        支持主题感知的概括生成
        
        Args:
            successful_docs: 成功处理的文档列表
            topic: 研究主题，用于生成有针对性的概括
            
        Returns:
            概括文本
        """
        summary_parts = []
        
        for doc in successful_docs:
            doc_info = doc['summary']
            title = doc['title']
            
            summary_text = f"📄 《{title}》\n"
            
            # 添加元数据
            metadata = doc_info['metadata']
            summary_text += f"   📁 文件: {metadata['filename']}\n"
            summary_text += f"   📊 文件大小: {metadata['file_size']} 字节\n"
            
            if metadata.get('num_pages'):
                summary_text += f"   📖 页数: {metadata['num_pages']}\n"
            
            # 添加增强概括（如果有）
            if 'enhanced_summary' in doc_info:
                summary_text += f"   📝 学术概括: {doc_info['enhanced_summary']}\n"
                
                # 如果有主题，添加主题相关性说明
                if topic:
                    summary_text += f"   🎯 与主题 '{topic}' 的相关性:\n"
                    summary_text += f"   该文档为研究主题提供了重要的理论基础和实践指导。\n"
            else:
                # 使用原始概括的一部分
                original_summary = doc_info['summary']
                if len(original_summary) > 300:
                    original_summary = original_summary[:300] + "..."
                summary_text += f"   📝 内容概括: {original_summary}\n"
                
                # 如果有主题，添加主题相关性说明
                if topic:
                    summary_text += f"   🎯 与主题 '{topic}' 的相关性:\n"
                    summary_text += f"   该文档与研究主题相关，提供了有价值的参考信息。\n"
            
            summary_parts.append(summary_text)
        
        return "\n".join(summary_parts)

class GeneratorAgent:
    """
    生成Agent：负责基于查找结果生成最终文献综述
    """
    
    def __init__(self, llm_client, report_generator=None):
        """
        初始化生成Agent
        
        Args:
            llm_client: LLMClient实例
            report_generator: ReportGenerator实例
        """
        self.llm_client = llm_client
        self.report_generator = report_generator
        
    def _extract_citations_from_context(self, search_context):
        """
        从搜索上下文中提取所有文献引用信息
        
        Args:
            search_context: 搜索上下文
            
        Returns:
            文献引用列表
        """
        citations = []
        
        # 使用正则表达式提取引用信息
        import re
        
        # 匹配引用格式
        citation_pattern = r'📖 引用: (.+?)(?=\n|$)'
        matches = re.findall(citation_pattern, search_context)
        
        for match in matches:
            citation = match.strip()
            if citation and citation not in citations:
                citations.append(citation)
        
        return citations
    
    def _generate_latex_bibliography(self, citations):
        """
        生成LaTeX格式的参考文献列表
        
        Args:
            citations: 引用列表
            
        Returns:
            LaTeX格式的参考文献
        """
        if not citations:
            return ""
        
        bibliography_items = []
        
        for i, citation in enumerate(citations, 1):
            # 简化引用格式，移除特殊字符
            clean_citation = citation.replace('&', '\\&').replace('%', '\\%').replace('#', '\\#')
            bibliography_items.append(f"\\item[{i}] {clean_citation}")
        
        bibliography = """
\\begin{thebibliography}{" + str(len(citations)) + f"}

{chr(10).join(bibliography_items)}

\\end{thebibliography}"""
        
        return bibliography
    
    def generate_literature_review_latex(self, topic, search_context):
        """
        生成LaTeX格式的文献综述
        
        Args:
            topic: 研究主题
            search_context: 查找Agent提供的上下文
            
        Returns:
            LaTeX格式的文献综述
        """
        logger.info("开始生成LaTeX格式文献综述...")
        
        # 记录Agent开始工作
        if self.report_generator:
            self.report_generator.record_agent_start("generator_agent")
        
        try:
            # 提取引用信息
            citations = self._extract_citations_from_context(search_context)
            bibliography = self._generate_latex_bibliography(citations)
            
            # 构建LaTeX文献综述生成提示词
            prompt = f"""
你是一个专业的学术文献综述撰写专家，精通LaTeX格式。请基于以下知识图谱中检索到的研究路径和相关文献，为指定主题撰写一篇高质量的LaTeX格式文献综述。

研究主题：{topic}

检索到的研究路径和文献信息：
{search_context}

请按照以下LaTeX格式结构撰写文献综述：

```latex
\\documentclass[a4paper,12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{ctex}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{graphicx}}
\\usepackage{{url}}
\\usepackage{{hyperref}}
\\usepackage{{natbib}}
\\usepackage{{geometry}}
\\geometry{{a4paper,left=2.5cm,right=2.5cm,top=2.5cm,bottom=2.5cm}}

\\title{{{topic}的文献综述}}
\\author{{学术文献综述生成器}}
\\date{{\\today}}

\\begin{{document}}

\\maketitle

\\section{{引言}}
% 研究背景和意义
% 研究主题的重要性
% 综述的目的和范围

\\section{{研究现状分析}}
% 基于检索到的研究路径，分析当前研究的主要方向
% 总结不同研究路径的特点和贡献
% 识别研究热点和发展趋势

\\section{{核心概念与理论框架}}
% 系统梳理研究主题的核心概念
% 分析概念间的关联关系
% 构建理论框架

\\section{{文献评述}}
% 基于检索到的文献，评述重要研究成果
% 分析不同文献的贡献和局限性
% 比较不同研究方法的优缺点
% 在适当位置使用\\cite{{ref1}}等格式引用文献

\\section{{研究展望}}
% 指出当前研究的空白和不足
% 提出未来研究的可能方向
% 总结综述的主要结论

{bibliography}

\\end{{document}}
```

撰写要求：
- 内容要基于提供的检索信息，不要凭空捏造
- 逻辑清晰，层次分明
- 语言学术化、专业化
- 字数控制在2000-3000字
- 在适当位置使用\\cite{{ref1}}、\\cite{{ref2}}等格式引用文献
- 体现批判性思维和学术深度
- 确保LaTeX语法正确，可以直接编译
- 使用自然通顺的语句引用检索到的信息，避免直接出现路径或节点

请生成完整的LaTeX文献综述代码：
"""
            
            # 调用大模型生成LaTeX文献综述
            response = self.llm_client.call_llm(prompt)
            
            # 清理和验证LaTeX代码
            latex_code = self._clean_latex_code(response)
            
            logger.info("LaTeX文献综述生成完成")
            
            # 记录Agent成功完成工作
            if self.report_generator:
                gen_output_data = {
                    'topic': topic,
                    'output_format': 'LaTeX',
                    'citations_count': len(citations),
                    'latex_code_length': len(latex_code),
                    'citations': citations
                }
                self.report_generator.record_agent_end("generator_agent", gen_output_data)
            
            return latex_code
            
        except Exception as e:
            # 记录错误
            if self.report_generator:
                self.report_generator.record_error("generator_agent", str(e))
            logger.error(f"生成LaTeX文献综述时出错: {str(e)}")
            return f"生成LaTeX文献综述时出错: {str(e)}"
    
    def _clean_latex_code(self, latex_code):
        """
        清理和验证LaTeX代码
        
        Args:
            latex_code: 原始LaTeX代码
            
        Returns:
            清理后的LaTeX代码
        """
        try:
            # 移除代码块标记
            latex_code = latex_code.replace('```latex', '').replace('```', '')
            
            # 确保文档结构完整
            if not latex_code.strip().startswith('\\documentclass'):
                latex_code = '\\documentclass[a4paper,12pt]{article}\n' + latex_code
            
            if '\\begin{document}' not in latex_code:
                latex_code = latex_code.replace('\\maketitle', '\\begin{document}\n\\maketitle')
            
            if '\\end{document}' not in latex_code:
                latex_code += '\n\\end{document}'
            
            return latex_code.strip()
            
        except Exception as e:
            logger.warning(f"清理LaTeX代码时出错: {str(e)}")
            return latex_code

    def generate_literature_review(self, topic, search_context, use_latex=False):
        """
        生成文献综述
        
        Args:
            topic: 研究主题
            search_context: 查找Agent提供的上下文
            use_latex: 是否使用LaTeX格式输出
            
        Returns:
            生成的文献综述
        """
        # 记录Agent开始工作
        if self.report_generator:
            self.report_generator.record_agent_start("generator_agent")
        
        logger.info("开始生成文献综述...")
        
        try:
            if use_latex:
                return self.generate_literature_review_latex(topic, search_context)
            else:
                # 构建文献综述生成提示词
                prompt = f"""
你是一个专业的学术文献综述撰写专家。请基于以下知识图谱中检索到的研究路径和相关文献，为指定主题撰写一篇高质量的文献综述。

研究主题：{topic}

检索到的研究路径和文献信息：
{search_context}

请按照以下结构撰写文献综述：

1. **引言**
   - 研究背景和意义
   - 研究主题的重要性
   - 综述的目的和范围

2. **研究现状分析**
   - 基于检索到的研究路径，分析当前研究的主要方向
   - 总结不同研究路径的特点和贡献
   - 识别研究热点和发展趋势

3. **核心概念与理论框架**
   - 系统梳理研究主题的核心概念
   - 分析概念间的关联关系
   - 构建理论框架

4. **文献评述**
   - 基于检索到的文献，评述重要研究成果
   - 分析不同文献的贡献和局限性
   - 比较不同研究方法的优缺点

5. **研究展望**
   - 指出当前研究的空白和不足
   - 提出未来研究的可能方向
   - 总结综述的主要结论

撰写要求：
- 内容要基于提供的检索信息，不要凭空捏造
- 逻辑清晰，层次分明
- 语言学术化、专业化
- 字数控制在2000-3000字
- 在适当位置引用检索到的文献信息
- 体现批判性思维和学术深度
- 使用自然通顺的语句引用检索到的信息，避免直接出现路径或节点，因为用户不了解这些细节

请开始撰写文献综述：
"""
            
                # 调用大模型生成文献综述
                review = self.llm_client.call_llm(prompt)
                
                logger.info("文献综述生成完成")
                
                # 记录Agent成功完成工作
                if self.report_generator:
                    gen_output_data = {
                        'topic': topic,
                        'output_format': '普通文本',
                        'review_length': len(review),
                        'word_count': len(review.split())
                    }
                    self.report_generator.record_agent_end("generator_agent", gen_output_data)
                
                return review
            
        except Exception as e:
            # 记录错误
            if self.report_generator:
                self.report_generator.record_error("generator_agent", str(e))
            error_msg = f"生成文献综述时出错: {str(e)}"
            logger.error(error_msg)
            return error_msg

class LiteratureReviewGenerator:
    """
    文献综述生成器主类，协调多个Agent的工作
    """
    
    def __init__(self, generate_reports=True):
        """
        初始化文献综述生成器
        
        Args:
            generate_reports: 是否生成报告文件，True生成，False不生成
        """
        # 初始化知识图谱
        self.kg = Neo4jKnowledgeGraph(
            NEO4J_CONFIG["uri"],
            NEO4J_CONFIG["user"],
            NEO4J_CONFIG["password"]
        )
        
        # 使用llm.py的配置读取功能
        self.llm_client = LLMClient()
        
        # 初始化DocumentManager
        self.document_manager = DocumentManager()
        
        # 根据参数决定是否初始化报告生成器
        self.generate_reports = generate_reports
        self.report_generator = ReportGenerator() if generate_reports else None
        
        # 初始化各个Agent
        self.search_agent = SearchAgent(self.kg, llm_client=self.llm_client, report_generator=self.report_generator)
        self.document_reader_agent = DocumentReaderAgent(
            document_manager=self.document_manager,
            llm_client=self.llm_client,
            report_generator=self.report_generator
        )
        self.generator_agent = GeneratorAgent(self.llm_client, report_generator=self.report_generator)
        
        logger.info(f"文献综述生成器初始化完成（报告生成功能: {'开启' if generate_reports else '关闭'}")
    
    def generate_review(self, topic, use_latex=False, max_documents=10, prompt_supplement="", max_workers=4):
        """
        生成指定主题的文献综述
        使用三阶段流程：搜索 -> 选择最有价值文献 -> 文档阅读 -> 生成
        
        Args:
            topic: 研究主题
            use_latex: 是否使用LaTeX格式输出
            max_documents: 最多返回的最有价值文献数量
            prompt_supplement: 补充提示词，传递给GeneratorAgent
            max_workers: 并行处理文档的最大工作线程数
            
        Returns:
            生成的文献综述
        """
        # 根据配置决定是否启动报告会话
        if self.generate_reports and self.report_generator:
            self.report_generator.start_session(topic)
        
        logger.info(f"开始为主题 '{topic}' 生成文献综述（四阶段流程）...")
        
        try:
            # 第一阶段：查找Agent搜索相关路径
            logger.info("第一阶段：启动查找Agent搜索相关研究路径...")
            search_paths = self.search_agent.find_multiple_paths(topic)
            
            if not search_paths:
                error_msg = f"未能为主题 '{topic}' 找到相关的研究路径和文献信息，无法生成文献综述。"
                if self.generate_reports and self.report_generator:
                    self.report_generator.record_error(error_msg)
                return error_msg
            
            # 根据配置决定是否保存搜索阶段报告
            if self.generate_reports and self.report_generator:
                search_stage_file = self.report_generator.save_stage_report(
                    "search_completed", 
                    {"search_paths": search_paths, "path_count": len(search_paths)}
                )
                logger.info(f"搜索阶段报告已保存: {search_stage_file}")
            
            # 新增阶段：选择最有价值的k篇文献
            logger.info(f"第二阶段：从搜索结果中选择最有价值的{max_documents}篇文献...")
            all_documents, selected_documents = self.search_agent.select_top_documents_from_paths(
                search_paths, topic, max_documents
            )
            logger.info(f"文档选择完成：从{len(all_documents)}篇文献中选择了{len(selected_documents)}篇最有价值的文献")
            logger.info(f"选中文献列表：{[doc['filename'] for doc in selected_documents]}")
            
            # 根据配置决定是否保存文献选择阶段报告
            if self.generate_reports and self.report_generator:
                selection_stage_file = self.report_generator.save_stage_report(
                    "documents_selected", 
                    {
                        "total_documents": len(all_documents),
                        "selected_documents": len(selected_documents),
                        "selected_titles": [doc['title'] for doc in selected_documents]
                    }
                )
                logger.info(f"文献选择阶段报告已保存: {selection_stage_file}")
            
            # 第三阶段：文档阅读Agent仅处理被选中的文献
            logger.info(f"第三阶段：启动文档阅读Agent处理已选择的{len(selected_documents)}篇文献...")
            selected_filenames = [doc['filename'] for doc in selected_documents]
            logger.info(f"传递给DocumentReaderAgent的选中文献列表：{selected_filenames}")
            processed_paths = self.document_reader_agent.process_documents_from_paths(
                search_paths, 
                topic=topic,
                selected_documents=selected_filenames,  # 只处理选中的文献
                max_workers=max_workers  # 使用指定数量的工作线程并行处理文档
            )
            
            # 更新搜索Agent的路径信息，包含文档处理结果
            self.search_agent.search_paths = processed_paths
            
            # 根据配置决定是否保存文档处理阶段报告
            if self.generate_reports and self.report_generator:
                documents_stage_file = self.report_generator.save_stage_report(
                    "documents_processed", 
                    {"processed_paths": processed_paths, "path_count": len(processed_paths)}
                )
                logger.info(f"文档处理阶段报告已保存: {documents_stage_file}")
            
            # 格式化包含文档概括的搜索结果
            search_context = self.search_agent.format_search_results(topic, max_documents)
            logger.info(f"已格式化选中的最有价值文献")
            
            # 第四阶段：生成Agent基于搜索结果生成文献综述
            logger.info(f"第四阶段：启动生成Agent撰写{'LaTeX格式' if use_latex else '普通格式'}文献综述...")
            # 如果有补充提示词，添加到search_context中
            if prompt_supplement:
                search_context = f"{search_context}\n\n额外要求：{prompt_supplement}"
            
            literature_review = self.generator_agent.generate_literature_review(topic, search_context, use_latex)
            
            # 根据配置决定是否保存综述生成阶段报告
            if self.generate_reports and self.report_generator:
                review_stage_file = self.report_generator.save_stage_report(
                    "review_generated", 
                    {"use_latex": use_latex, "max_documents": max_documents, "has_prompt_supplement": bool(prompt_supplement)}
                )
                logger.info(f"综述生成阶段报告已保存: {review_stage_file}")
                
                # 结束报告会话并保存最终完整报告
                self.report_generator.end_session()
                final_report_file = self.report_generator.save_report()
                
                if final_report_file:
                    logger.info(f"最终完整工作报告已保存到: {final_report_file}")
                
                logger.info(f"文献综述生成完成，主题：{topic}")
                logger.info(f"已生成5个报告文件：搜索阶段、文献选择阶段、文档处理阶段、综述生成阶段和最终完整报告")
            else:
                logger.info(f"文献综述生成完成，主题：{topic}")
                logger.info(f"报告生成功能已关闭")
                
            return literature_review
            
        except Exception as e:
            error_msg = f"生成文献综述时出错: {str(e)}"
            logger.error(error_msg)
            if self.generate_reports and self.report_generator:
                self.report_generator.record_error(error_msg)
                # 即使出错也要保存报告
                self.report_generator.end_session()
                self.report_generator.save_report()
            import traceback
            traceback.print_exc()
            
            return error_msg
        
        finally:
            # 关闭数据库连接
            self.kg.close()
            logger.info("数据库连接已关闭")
    
    def close(self):
        """关闭资源"""
        if hasattr(self, 'kg'):
            self.kg.close()

def main():
    """
    主函数：交互式文献综述生成
    使用三阶段多Agent协作模式：搜索Agent -> 文档阅读Agent -> 生成Agent
    """
    print("=" * 60)
    print("📚 基于知识图谱的文献综述生成器")
    print("=" * 60)
    print("🔍 使用三阶段多Agent协作模式：")
    print("   1️⃣ 搜索Agent：查找相关研究路径和文献")
    print("   2️⃣ 文档阅读Agent：阅读和概括检索到的文献")
    print("   3️⃣ 生成Agent：基于处理后的信息生成综述")
    print("   为您生成高质量的学术文献综述")
    print("   🆕 支持LaTeX格式输出，规范化文献引用")
    print("   📊 新增工作报告功能，记录每个Agent的详细输出")
    print()
    
    generator = LiteratureReviewGenerator()
    
    try:
        while True:
            topic = input("请输入研究主题（输入 'quit' 退出）: ").strip()
            
            if topic.lower() in ['quit', 'exit', '退出']:
                print("感谢使用文献综述生成器！")
                break
            
            if not topic:
                print("请输入有效的研究主题。")
                continue
            
            # 询问输出格式
            format_choice = input("选择输出格式（1: 普通文本格式，2: LaTeX格式，默认: 1）: ").strip()
            use_latex = format_choice == '2'
            
            format_desc = "LaTeX格式" if use_latex else "普通文本格式"
            print(f"\n🔍 正在为主题 '{topic}' 生成{format_desc}文献综述...")
            print("这可能需要几分钟时间，请耐心等待...")
            print("📊 将自动生成详细的工作报告...")
            print()
            
            start_time = time.time()
            
            # 生成文献综述
            review = generator.generate_review(topic, use_latex)
            
            end_time = time.time()
            duration = end_time - start_time
            
            print("\n" + "=" * 60)
            print(f"✅ {format_desc}文献综述生成完成（耗时: {duration:.2f}秒）")
            print("📊 详细工作报告已保存到 reports/ 目录")
            print("=" * 60)
            print()
            print(review)
            print("\n" + "=" * 60)
            
            # 如果是LaTeX格式，自动保存为.tex文件并尝试编译为PDF
            if use_latex:
                # 生成文件名，使用主题的前20个字符作为文件名，避免文件名过长
                safe_topic = re.sub(r'[\\/:*?"<>|]', '_', topic[:20])
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{safe_topic}_{timestamp}"
                tex_path = os.path.join("reports", f"{filename}.tex")
                pdf_path = os.path.join("reports", f"{filename}.pdf")
                
                # 确保reports目录存在
                os.makedirs("reports", exist_ok=True)
                
                # 保存LaTeX内容到.tex文件
                with open(tex_path, 'w', encoding='utf-8') as f:
                    f.write(review)
                
                print("\n💾 已自动保存LaTeX文件：")
                print(f"- LaTeX源文件: {tex_path}")
                
                # 尝试使用pdflatex编译生成PDF
                print("\n🔄 尝试编译PDF文件...")
                try:
                    # 检查pdflatex是否可用
                    subprocess.run(["pdflatex", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                    
                    # 编译LaTeX文件
                    result = subprocess.run(["pdflatex", "-output-directory=reports", tex_path], 
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if result.returncode == 0 and os.path.exists(pdf_path):
                        print("✅ PDF文件编译成功！")
                        print(f"- PDF文件: {pdf_path}")
                    else:
                        print("❌ PDF编译失败")
                        print("- 可能需要手动修复LaTeX代码中的错误")
                        print("- 推荐使用Overleaf在线LaTeX编辑器打开.tex文件")
                        
                except (subprocess.SubprocessError, FileNotFoundError):
                    print("⚠️  未检测到pdflatex命令")
                    print("- 请安装LaTeX发行版（如MiKTeX、TeX Live）")
                    print("- 或使用Overleaf在线LaTeX编辑器打开生成的.tex文件")
                    
                print()
            
            print("\n📋 工作报告说明：")
            print("- JSON格式：包含所有Agent的详细输出数据")
            print("- Markdown格式：包含简洁的会话总结")
            print("- 可用于分析生成过程和调试问题")
            print()
            
    except KeyboardInterrupt:
        print("\n\n程序被用户中断")
    except Exception as e:
        logger.error(f"程序运行出错: {str(e)}")
        print(f"程序运行出错: {str(e)}")
    finally:
        generator.close()

if __name__ == "__main__":
    main()