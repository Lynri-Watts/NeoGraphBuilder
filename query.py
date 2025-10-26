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
import configparser
from openai import OpenAI

# 导入数据库模块和LLM客户端
from database import Neo4jKnowledgeGraph
from llm import LLMClient

# 配置日志
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 从配置文件读取配置
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
config.read(config_path)

# 大模型配置
LLM_CONFIG = {
    "base_url": config.get('LLM_CONFIG', 'base_url'),
    "api_key": config.get('LLM_CONFIG', 'api_key'),
    "model": config.get('LLM_CONFIG', 'model')
}

# 大模型额外配置
LLM_EXTRA_CONFIG = {
    "hy_source": config.get('LLM_EXTRA_CONFIG', 'hy_source'),
    "hy_user": config.get('LLM_EXTRA_CONFIG', 'hy_user'),
    "agent_id": config.get('LLM_EXTRA_CONFIG', 'agent_id'),
    "should_remove_conversation": config.getboolean('LLM_EXTRA_CONFIG', 'should_remove_conversation')
}

# 数据库配置
NEO4J_CONFIG = {
    "uri": config.get('NEO4J_CONFIG', 'uri'),
    "user": config.get('NEO4J_CONFIG', 'user'),
    "password": config.get('NEO4J_CONFIG', 'password')
}

# RAG提示词模板
RAG_PROMPT_TEMPLATE = """
你是一个基于知识图谱的问答助手。请根据以下提供的相关信息，回答用户的问题。

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
        
        # 初始化LLM客户端
        self.llm_client = LLMClient(LLM_CONFIG, LLM_EXTRA_CONFIG)
    
    def get_embedding(self, text):
        """
        获取文本的嵌入向量
        
        Args:
            text: 输入文本
            
        Returns:
            numpy数组形式的嵌入向量
        """
        return self.kg.model.encode(text)
    
    def retrieve_relevant_info(self, query, top_k=5):
        """
        从知识图谱中检索与查询相关的信息
        
        Args:
            query: 用户查询
            top_k: 返回的最大相关项数
            
        Returns:
            包含相关节点和关系的上下文文本
        """
        logger.info(f"检索与查询 '{query}' 相关的信息...")
        
        # 使用database.py中的search_knowledge_by_text方法
        # 该方法已经封装了向量检索和关系获取的逻辑
        search_results = self.kg.search_knowledge_by_text(query, top_k)
        
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
            context_parts.append(f"相似度: {result['similarity']:.4f}")
            
            # 添加关系信息
            if result.get('relations'):
                context_parts.append("关系:")
                for rel in result['relations'][:3]:  # 最多显示3个关系
                    related_type_cn = "概念" if rel['related_node_type'] == 'Concept' else "实体"
                    context_parts.append(f"  - 与[{rel['related_name']}]({related_type_cn})的关系: {rel['relation_type']}")
        
        context = "\n".join(context_parts)
        
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