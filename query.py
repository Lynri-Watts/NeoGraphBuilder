#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于知识图谱的RAG问答脚本

该脚本实现：
1. 接收用户查询
2. 将查询转换为向量
3. 从Neo4j知识图谱中检索相关信息
4. 调用大模型生成基于检索内容的回答
"""

import os
import json
import logging
import time
import re
from collections import defaultdict
from openai import OpenAI

# 导入数据库模块和LLM客户端
from database import Neo4jKnowledgeGraph
from llm import LLMClient

# 配置日志
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# RAG提示词模板
RAG_PROMPT_TEMPLATE = """
你是一个基于知识图谱的问答助手。请根据以下提供的相关信息，回答用户的问题。回答问题时，应当将相关信息与用户问题结合，考虑到信息的准确性和完整性。
同时，当你的回答涉及到知识图谱中的知识时，应当用通顺的语句复述这些知识，因为用户一般不能直接看到知识图谱的具体结构和关系。

相关信息：
{context}

用户问题：
{query}

请使用中文回答，确保回答准确、全面、简洁。如果信息不足，请明确表示无法回答相关部分，并基于已知信息尽可能提供帮助。
"""

class KnowledgeGraphQA:
    """
    基于知识图谱的问答类
    """
    
    def __init__(self):
        """
        初始化问答系统，连接数据库和模型
        """
        # 初始化Neo4j连接
        self.kg = Neo4jKnowledgeGraph(
            NEO4J_CONFIG["uri"],
            NEO4J_CONFIG["user"],
            NEO4J_CONFIG["password"]
        )
        
        # 使用llm.py的配置读取功能
        self.llm_client = LLMClient()
    
    def encode_text(self, text):
        """
        使用sentence-transformers编码文本
        
        Args:
            text: 输入文本
            
        Returns:
            numpy数组形式的嵌入向量
        """
        import time
        start_time = time.time()
        
        # 记录编码前的详细信息
        text_length = len(text)
        text_preview = text[:100] + "..." if text_length > 100 else text
        logger.info(f"[SENTENCE_TRANSFORMER] query.py encode_text开始 - 长度: {text_length}, 预览: '{text_preview}'")
        
        try:
            encode_start = time.time()
            vector = self.kg.model.encode(text)
            encode_time = time.time() - encode_start
            
            # 记录编码后的信息
            logger.info(f"[SENTENCE_TRANSFORMER] query.py encode_text完成 - 耗时: {encode_time:.3f}s, 向量维度: {vector.shape if hasattr(vector, 'shape') else len(vector)}")
            
            total_time = time.time() - start_time
            logger.info(f"[SENTENCE_TRANSFORMER] query.py encode_text总耗时: {total_time:.3f}s")
            
            return vector
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"[SENTENCE_TRANSFORMER] query.py encode_text出错 - 耗时: {error_time:.3f}s, 文本: '{text[:50]}...', 错误: {str(e)}")
            # 返回零向量作为fallback
            import numpy as np
            return np.zeros(384)
    
    def get_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            numpy数组形式的嵌入向量
        """
        import time
        start_time = time.time()
        
        # 记录编码前的详细信息
        text_length = len(text)
        text_preview = text[:100] + "..." if text_length > 100 else text
        logger.info(f"[SENTENCE_TRANSFORMER] query.py get_embedding开始 - 长度: {text_length}, 预览: '{text_preview}'")
        
        try:
            encode_start = time.time()
            vector = self.kg.model.encode(text)
            encode_time = time.time() - encode_start
            
            # 记录编码后的信息
            logger.info(f"[SENTENCE_TRANSFORMER] query.py get_embedding完成 - 耗时: {encode_time:.3f}s, 向量维度: {vector.shape if hasattr(vector, 'shape') else len(vector)}")
            
            total_time = time.time() - start_time
            logger.info(f"[SENTENCE_TRANSFORMER] query.py get_embedding总耗时: {total_time:.3f}s")
            
            return vector
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"[SENTENCE_TRANSFORMER] query.py get_embedding出错 - 耗时: {error_time:.3f}s, 文本: '{text[:50]}...', 错误: {str(e)}")
            # 返回零向量作为fallback
            import numpy as np
            return np.zeros(384)
    
    def extract_keywords(self, query):
        """
        从用户查询中提取关键词
        
        Args:
            query: 用户查询文本
            
        Returns:
            关键词列表
        """
        # 移除标点符号，保留中文字符、英文字符和数字
        cleaned_query = re.sub(r'[^\w\s\u4e00-\u9fff]', ' ', query)
        
        # 按空格分割，过滤空字符串和单字符
        words = [word.strip() for word in cleaned_query.split() if len(word.strip()) > 1]
        
        # 如果没有有效关键词，使用原查询
        if not words:
            words = [query]
        
        # 去重并保持顺序
        seen = set()
        keywords = []
        for word in words:
            if word not in seen:
                seen.add(word)
                keywords.append(word)
        
        logger.info(f"提取的关键词: {keywords}")
        return keywords
    
    def merge_search_results(self, all_results, keywords):
        """
        合并多个关键词的搜索结果
        
        Args:
            all_results: 所有关键词的搜索结果列表
            keywords: 关键词列表
            
        Returns:
            合并并去重后的搜索结果
        """
        # 使用字典去重，以节点名称和类型为键
        merged_results = {}
        
        for keyword_results in all_results:
            for result in keyword_results:
                # 创建唯一键
                key = (result['name'], result['node_type'])
                
                # 如果节点已存在，更新信息
                if key in merged_results:
                    existing = merged_results[key]
                    # 保留更高的相似度分数
                    if result['similarity'] > existing['similarity']:
                        existing['similarity'] = result['similarity']
                    
                    # 合并关系信息
                    if result.get('relations'):
                        if not existing.get('relations'):
                            existing['relations'] = []
                        
                        # 添加新的关系，避免重复
                        existing_relations = set((rel['related_name'], rel['relation_type']) for rel in existing['relations'])
                        for rel in result['relations']:
                            rel_key = (rel['related_name'], rel['relation_type'])
                            if rel_key not in existing_relations:
                                existing['relations'].append(rel)
                else:
                    # 新节点，直接添加
                    merged_results[key] = result.copy()
        
        # 转换为列表并按相似度排序
        final_results = list(merged_results.values())
        final_results.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"合并后的结果数量: {len(final_results)}")
        return final_results
    
    def retrieve_relevant_info(self, query, top_k=5):
        """
        从知识图谱中检索与查询相关的信息
        先将查询拆分为关键词，再逐一检索，最后合并结果
        
        Args:
            query: 用户查询
            top_k: 返回的最大相关项数
            
        Returns:
            包含相关节点和关系的上下文文本
        """
        logger.info(f"检索与查询 '{query}' 相关的信息...")
        
        # 提取关键词
        keywords = self.extract_keywords(query)
        
        # 如果只有一个关键词且就是原查询，使用原有逻辑
        if len(keywords) == 1 and keywords[0] == query:
            search_results = self.kg.search_knowledge_by_text(query, top_k)
        else:
            # 对每个关键词进行检索
            all_results = []
            for keyword in keywords:
                logger.info(f"检索关键词: '{keyword}'")
                keyword_results = self.kg.search_knowledge_by_text(keyword, top_k)
                all_results.append(keyword_results)
            
            # 合并结果
            search_results = self.merge_search_results(all_results, keywords)
            
            # 限制最终结果数量
            search_results = search_results[:top_k]
        
        # 构建上下文
        context_parts = []
        
        for i, result in enumerate(search_results):
            context_parts.append(f"\n### 相关{i+1}###")
            # 转换节点类型显示
            node_type_cn = "概念" if result['node_type'] == 'Concept' else "实体"
            context_parts.append(f"类型: {node_type_cn}")
            context_parts.append(f"名称: {result['name']}")
            if result.get('description'):
                context_parts.append(f"描述: {result['description']}")
            # context_parts.append(f"相似度: {result['similarity']:.4f}")
            
            # 添加关系信息
            if result.get('relations'):
                context_parts.append("关系:")
                for rel in result['relations'][:3]:  # 最多显示3个关系
                    related_type_cn = "概念" if rel['related_node_type'] == 'Concept' else "实体"
                    context_parts.append(f"  - 与[{rel['related_name']}]({related_type_cn})的关系: {rel['relation_type']}")
        
        context = "\n".join(context_parts)

        logger.info(f"检索到的上下文:\n{context}")
        
        # 如果没有检索到相关信息，生成一个提示
        if not search_results:
            context = "未找到与查询直接相关的知识图谱信息。请尝试调整问题或使用更具体的关键词。"
            logger.warning("未找到相关信息")
        else:
            logger.info(f"检索完成，获取了 {len(search_results)} 条相关信息")
        
        return context
    
    def generate_answer(self, query, context):
        """
        调用大模型生成回答
        
        Args:
            query: 用户查询
            context: 检索到的相关信息
            
        Returns:
            生成的回答文本
        """
        logger.info("调用大模型生成回答...")
        
        # 构建提示词
        prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
        
        # 使用LLMClient生成回答
        answer = self.llm_client.call_llm(prompt)
        
        return answer
    
    def answer_query(self, query):
        """
        回答用户查询的完整流程
        
        Args:
            query: 用户查询
            
        Returns:
            回答文本
        """
        try:
            start_time = time.time()
            
            # 检索相关信息
            context = self.retrieve_relevant_info(query)
            
            # 生成回答
            answer = self.generate_answer(query, context)
            
            elapsed_time = time.time() - start_time
            logger.info(f"问答完成，耗时: {elapsed_time:.2f}秒")
            
            return answer
            
        except Exception as e:
            logger.error(f"问答过程中发生错误: {str(e)}")
            raise
    
    def close(self):
        """
        关闭数据库连接
        """
        if self.kg:
            self.kg.close()

def main():
    """
    主函数，启动交互式问答
    """
    print("=== 知识图谱RAG问答系统 ===")
    print("请输入您的问题，输入'exit'或'quit'退出")
    
    # 初始化问答系统
    qa_system = KnowledgeGraphQA()
    
    try:
        while True:
            # 获取用户输入
            query = input("\n问题: ").strip()
            
            # 检查退出命令
            if query.lower() in ['exit', 'quit', '退出', '结束']:
                print("感谢使用，再见！")
                break
            
            # 回答问题
            try:
                answer = qa_system.answer_query(query)
                print("\n回答:")
                print(answer)
            except Exception as e:
                print(f"回答过程中出现错误: {str(e)}")
    
    finally:
        # 关闭连接
        qa_system.close()

if __name__ == "__main__":
    main()