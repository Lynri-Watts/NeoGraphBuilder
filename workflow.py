#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于LLM的知识图谱构建完整工作流

该脚本实现：
1. 从文本文件读取输入内容
2. 调用大模型提取概念和关系
3. 解析结构化数据
4. 将概念和关系插入Neo4j数据库
"""

from ast import parse
import os
import json
import logging
import time
import configparser
from openai import OpenAI
from llm import LLMClient

# 导入数据库模块
from database import Neo4jKnowledgeGraph

# 配置日志 - 仅在workflow.py中配置一次
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

# 系统提示词 - 要求输出nodes和relations结构化数据
system_prompt = """
你现在的任务是从文本中提取知识图谱所需的结构化信息，包括节点(nodes)和关系(relations)。

请严格按照以下JSON格式输出，不要包含任何其他内容：
{
  "nodes": [
    {
      "name": "节点名称",
      "type": "Concept",  // 或 "Entity"
      "description": "节点的详细描述",
      "context": ["相关上下文词汇1", "相关上下文词汇2"],
      "source_document": "来源文档标识",
      "aliases": ["别称1", "缩写1", "别称2"]
    }
    // 更多节点...
  ],
  "relations": [
    {
      "source": "源节点名称",
      "target": "目标节点名称",
      "type": "关系类型",
      "weight": 1.0,
      "properties": {
        "描述": "关系的详细描述"
      }
    }
    // 更多关系...
  ]
}

提取要求：
1. 节点(nodes)列表应包含文本中所有重要的概念和实体
2. 对于每个节点，请正确设置type字段：
   - "Concept"：表示抽象的概念、理论、思想、学科等非实体事物，或者实体所属的类别（如：手机、电脑、机器学习、自动社交编辑）
   - "Entity"：表示现实中唯一存在的具体事物，必须满足以下条件：
     * 不是类别或集合
     * 不存在歧义
     * 能够脱离任何语境单独存在
     * 合法实体示例：2017年、iPhone 15、微软公司、北京市
     * 非法实体示例：本文、笔者（依赖特定语境）、9月3日、David（存在歧义）、Iphone全球使用统计（不是单一存在）
3. 每个节点需提供name(名称)、type(类型)和description(描述)，context(上下文)和source_document(来源)为可选
4. 每个节点必须包含aliases字段，列出该节点的所有可能别称和缩写形式，即使文章中未直接提及
5. 无论输入为何种语言，无论原文中是否出现，请优先使用中文全称作为name字段的第一标识符
6. 关系(relations)列表应包含节点之间的所有有意义的连接，关系类型请使用中文全称
7. 关系必须包含source(源)、target(目标)和type(类型)，weight(权重)默认为1.0，properties(属性)为可选
8. 请确保输出的JSON格式完全正确，可被程序直接解析
"""

system_prompt_simplified = """
你现在的任务是从文本中提取知识图谱所需的结构化信息，包括节点(nodes)和关系(relations)。

请严格按照以下JSON格式输出，不要包含任何其他内容：
{
  "nodes": [
    {
      "name": "节点名称",
      "type": "Concept",  // 或 "Entity"
      "aliases": ["别称1", "缩写1", "别称2"]
    }
    // 更多节点...
  ],
  "relations": [
    {
      "source": "源节点名称",
      "target": "目标节点名称",
      "type": "关系类型",
      "weight" : 1.0
    }
    // 更多关系...
  ]
}

提取要求：
1. 节点(nodes)列表应包含文本中所有重要的概念和实体
2. 对于每个节点，请正确设置type字段：
   - "Concept"：表示抽象的概念、理论、思想、学科等非实体事物，或者实体所属的类别（如：手机、电脑、机器学习、自动社交编辑）
   - "Entity"：表示现实中唯一存在的具体事物，必须满足以下条件：
     * 不是类别或集合
     * 不存在歧义
     * 能够脱离任何语境单独存在
     * 合法实体示例：2017年、iPhone 15、微软公司、北京市
     * 非法实体示例：本文、笔者（依赖特定语境）、9月3日、David（存在歧义）、Iphone全球使用统计（不是单一存在）
3. 每个节点需提供name(名称)、type(类型)
4. 每个节点必须包含aliases字段，列出该节点的所有可能别称和缩写形式，即使文章中未直接提及
5. 无论输入为何种语言，无论原文中是否出现，请优先使用中文全称作为name字段的第一标识符
6. 关系(relations)列表应包含节点之间的所有有意义的连接，关系类型请使用中文全称
7. 关系必须包含source(源)、target(目标)和type(类型)，weight(权重)默认为1.0
8. 请确保输出的JSON格式完全正确，可被程序直接解析
"""

def load_input_file(file_path):
    """从文件加载输入文本"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        logger.info(f"成功加载输入文件: {file_path}")
        return content
    except Exception as e:
        logger.error(f"加载输入文件失败: {str(e)}")
        raise

def call_llm(text, source_document="unknown"):
    """
    调用大模型提取概念和关系
    
    Args:
        text: 输入文本
        source_document: 来源文档标识
        
    Returns:
        dict: 包含concepts和relations的字典
    """
    # 构建完整的提示词
    prompt = f"<systemprompt>{system_prompt_simplified}</systemprompt>\n<userinput>文本内容：{text}\n来源文档：{source_document}</userinput>"
    
    # 使用LLMClient类处理大模型调用
    llm_client = LLMClient(LLM_CONFIG, LLM_EXTRA_CONFIG)
    return llm_client.call_llm(prompt)

def parse_kg_json_response(answer_str: str) -> dict[str, any]:
        """
        解析知识图谱JSON响应
        
        Args:
            answer_str: 模型返回的回答字符串
            
        Returns:
            解析后的知识图谱数据字典
        """
        try:
            # 清理可能的额外内容
            clean_answer = answer_str.strip()
            
            # 处理代码块格式
            if "```json" in clean_answer:
                clean_answer = clean_answer.split("```json")[1].split("```")[0]
            elif "```" in clean_answer:
                clean_answer = clean_answer.split("```")[1].split("```")[0]
            
            # 提取JSON部分
            if "{" in clean_answer:
                clean_answer = clean_answer[clean_answer.find("{"):]
            if "}" in clean_answer:
                clean_answer = clean_answer[:clean_answer.rfind("}")+1]
            
            logger.info(f"清理后的JSON数据前200字符: {clean_answer[:200]}...")
            
            data = json.loads(clean_answer)
            elapsed_time = time.time() - time.time()  # 重新计算
            logger.info(f"大模型调用完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"成功提取 {len(data.get('concepts', []))} 个概念和 {len(data.get('relations', []))} 个关系")
            
            return data
        except json.JSONDecodeError as e:
            logger.error(f"解析知识图谱JSON数据失败: {str(e)}")
            logger.error(f"完整回答内容: {answer_str}")
            # 尝试使用正则表达式提取JSON
            try:
                json_pattern = r'\{(?:[^\{\}]|(?R))*\}'
                matches = re.findall(json_pattern, answer_str)
                if matches:
                    logger.info(f"找到 {len(matches)} 个可能的JSON对象，尝试第一个")
                    data = json.loads(matches[0])
                    logger.info(f"使用正则表达式提取成功")
                    return data
                else:
                    logger.error("正则表达式未能找到有效JSON")
            except Exception as fallback_e:
                logger.error(f"备用解析方法也失败: {str(fallback_e)}")
            raise

def process_knowledge_graph(data, kg=None, source_document="unknown"):
    """
    处理知识图谱数据并插入数据库
    
    Args:
        data: 包含nodes和relations的字典
        kg: 已初始化的Neo4jKnowledgeGraph实例（可选）
        source_document: 来源文档标识
        
    Returns:
        dict: 处理结果统计
    """
    local_kg = kg
    try:
        # 如果没有提供kg实例，则创建一个
        if local_kg is None:
            # 初始化数据库连接
            logger.info("初始化Neo4j数据库连接...")
            local_kg = Neo4jKnowledgeGraph(
                NEO4J_CONFIG["uri"],
                NEO4J_CONFIG["user"],
                NEO4J_CONFIG["password"]
            )
        
        # 处理节点
        nodes = data.get('nodes', [])
        # 为每个节点添加source_document信息（如果未提供）
        for node in nodes:
            if 'source_document' not in node:
                node['source_document'] = source_document
            # 确保context字段存在
            if 'context' not in node:
                node['context'] = []
            # 确保type字段存在，默认为Concept
            if 'type' not in node:
                node['type'] = 'Concept'
        
        logger.info(f"开始插入 {len(nodes)} 个节点到数据库...")
        node_count = local_kg.insert_nodes(nodes)
        logger.info(f"成功插入 {node_count} 个节点")
        
        # 处理关系
        relations = data.get('relations', [])
        logger.info(f"开始插入 {len(relations)} 个关系到数据库...")
        relation_count = local_kg.insert_relations(relations)
        logger.info(f"成功插入 {relation_count} 个关系")
        
        return {
            "nodes_processed": node_count,
            "relations_processed": relation_count,
            "source_document": source_document
        }
        
    except Exception as e:
        logger.error(f"处理知识图谱数据失败: {str(e)}")
        raise
    finally:
        # 只在本地创建的实例才关闭连接
        if kg is None and local_kg:
            local_kg.close()
            logger.info("数据库连接已关闭")

def main(input_file=None, text=None, kg=None):
    """
    主函数 - 知识图谱构建工作流
    
    Args:
        input_file: 输入文件路径
        text: 直接输入文本（如果不使用文件）
        kg: 已初始化的Neo4jKnowledgeGraph实例（可选）
    """
    try:
        # 加载输入
        if input_file:
            text = load_input_file(input_file)
            source_document = os.path.basename(input_file)
        else:
            source_document = "direct_input"
        
        if not text:
            raise ValueError("必须提供输入文本或文件路径")
        
        # 调用大模型提取知识
        kg_data = parse_kg_json_response(call_llm(text, source_document))
        
        # 处理并存储到数据库
        result = process_knowledge_graph(kg_data, kg, source_document)
        
        # 打印最终结果
        logger.info("=== 知识图谱构建完成 ===")
        logger.info(f"来源文档: {result['source_document']}")
        logger.info(f"处理节点数: {result['nodes_processed']}")
        logger.info(f"处理关系数: {result['relations_processed']}")
        
        return result
        
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 示例用法
    # 从文件处理
    main(input_file="userinput_半结构化文本.txt")