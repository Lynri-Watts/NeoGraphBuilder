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

import os
import json
import logging
import time
from openai import OpenAI

# 导入数据库模块
from database import Neo4jKnowledgeGraph

# 配置日志 - 仅在workflow.py中配置一次
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 大模型配置
LLM_CONFIG = {
    "base_url": "http://localhost:8001/v1",
    "api_key": "nd0aaKhNvMmCDWTVsIt5bdzg/hB1Ry31IZtFQB2FcT1W7bzpM7OjgFomLgACTR46hUqZMK0jlcULvCUaYJvRGg==",
    "model": "deepseek-r1"
}

# 数据库配置
NEO4J_CONFIG = {
    "uri": "neo4j://127.0.0.1:7687",
    "user": "neo4j",
    "password": "12345678"
}

# 系统提示词 - 要求输出concepts和relations结构化数据
system_prompt = """
你现在的任务是从文本中提取知识图谱所需的结构化信息，包括概念(concepts)和关系(relations)。

请严格按照以下JSON格式输出，不要包含任何其他内容：
{
  "concepts": [
    {
      "name": "概念名称",
      "description": "概念的详细描述",
      "context": ["相关上下文词汇1", "相关上下文词汇2"],
      "source_document": "来源文档标识"
    }
    // 更多概念...
  ],
  "relations": [
    {
      "source": "源概念名称",
      "target": "目标概念名称",
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
1. 概念(concepts)列表应包含文本中所有重要实体、概念和主题
2. 每个概念需提供name(名称)和description(描述)，context(上下文)和source_document(来源)为可选
3. 关系(relations)列表应包含概念之间的所有有意义的连接
4. 关系必须包含source(源)、target(目标)和type(类型)，weight(权重)默认为1.0，properties(属性)为可选
5. 请确保输出的JSON格式完全正确，可被程序直接解析
6. 使用与输入文本相同的语言进行输出
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
    try:
        logger.info("开始调用大模型提取知识图谱数据...")
        start_time = time.time()
        
        client = OpenAI(
            base_url=LLM_CONFIG["base_url"],
            api_key=LLM_CONFIG["api_key"]
        )
        
        # 构建提示词，包含源文档信息
        prompt = f"<systemprompt>{system_prompt}</systemprompt>\n<userinput>文本内容：{text}\n来源文档：{source_document}</userinput>"
        
        response = client.chat.completions.create(
            model=LLM_CONFIG["model"],
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            extra_body={
                "hy_source": "web",
                "hy_user": "8fe2d96156694f6fac08fff143176d3e;",
                "agent_id": "naQivTmsDa",
                "should_remove_conversation": True
            }
        )
        
        # 处理流式响应 - 正确解析每个chunk的JSON
        think_str = ""
        answer_str = ""
        last_progress_length = 0  # 记录上一次显示进度时的长度
        progress_interval = 500  # 进度显示间隔（字符数）
        
        for chunk in response:
            content = chunk.choices[0].delta.content or ""
            if content:
                try:
                    # 解析每个chunk的JSON数据
                    chunk_data = json.loads(content)
                    # 区分思考过程和最终回答
                    if chunk_data.get("type") == "think":
                        think_str += chunk_data.get("content", "")
                    elif chunk_data.get("type") == "text":
                        answer_str += chunk_data.get("msg", "")
                    
                    # 实时打印处理进度 - 当累积长度超过进度间隔时显示
                    if len(answer_str) > last_progress_length + progress_interval:
                        logger.info(f"大模型响应处理中...已累积{len(answer_str)}字符的结构化数据")
                        last_progress_length = len(answer_str)
                except json.JSONDecodeError:
                    # 如果chunk不是有效的JSON，可能是直接的文本响应
                    logger.warning(f"收到非JSON格式的响应chunk: {content[:50]}...")
                    answer_str += content
                    
                    # 对于非JSON响应也显示进度
                    if len(answer_str) > last_progress_length + progress_interval:
                        logger.info(f"大模型响应处理中...已累积{len(answer_str)}字符的响应数据")
                        last_progress_length = len(answer_str)
        
        logger.info(f"思考过程: {think_str[:200]}..." if think_str else "没有思考过程")
        logger.info(f"最终回答内容前200字符: {answer_str[:200]}...")
        
        # 解析最终回答中的JSON知识图谱数据
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
            elapsed_time = time.time() - start_time
            logger.info(f"大模型调用完成，耗时: {elapsed_time:.2f}秒")
            logger.info(f"成功提取 {len(data.get('concepts', []))} 个概念和 {len(data.get('relations', []))} 个关系")
            
            return data
        except json.JSONDecodeError as e:
            logger.error(f"解析知识图谱JSON数据失败: {str(e)}")
            logger.error(f"完整回答内容: {answer_str}")
            # 尝试使用正则表达式提取JSON
            try:
                import re
                json_pattern = r'\{(?:[^{}]|(?R))*\}'
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
            
    except Exception as e:
        logger.error(f"大模型调用失败: {str(e)}")
        raise

def process_knowledge_graph(data, source_document="unknown"):
    """
    处理知识图谱数据并插入数据库
    
    Args:
        data: 包含concepts和relations的字典
        source_document: 来源文档标识
        
    Returns:
        dict: 处理结果统计
    """
    kg = None
    try:
        # 初始化数据库连接
        logger.info("初始化Neo4j数据库连接...")
        kg = Neo4jKnowledgeGraph(
            NEO4J_CONFIG["uri"],
            NEO4J_CONFIG["user"],
            NEO4J_CONFIG["password"]
        )
        
        # 处理概念
        concepts = data.get('concepts', [])
        # 为每个概念添加source_document信息（如果未提供）
        for concept in concepts:
            if 'source_document' not in concept:
                concept['source_document'] = source_document
            # 确保context字段存在
            if 'context' not in concept:
                concept['context'] = []
        
        logger.info(f"开始插入 {len(concepts)} 个概念到数据库...")
        concept_count = kg.insert_concepts(concepts)
        logger.info(f"成功插入 {concept_count} 个概念")
        
        # 处理关系
        relations = data.get('relations', [])
        logger.info(f"开始插入 {len(relations)} 个关系到数据库...")
        relation_count = kg.insert_relations(relations)
        logger.info(f"成功插入 {relation_count} 个关系")
        
        return {
            "concepts_processed": concept_count,
            "relations_processed": relation_count,
            "source_document": source_document
        }
        
    except Exception as e:
        logger.error(f"处理知识图谱数据失败: {str(e)}")
        raise
    finally:
        # 确保关闭数据库连接
        if kg:
            kg.close()
            logger.info("数据库连接已关闭")

def main(input_file=None, text=None):
    """
    主函数 - 知识图谱构建工作流
    
    Args:
        input_file: 输入文件路径
        text: 直接输入文本（如果不使用文件）
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
        kg_data = call_llm(text, source_document)
        
        # 处理并存储到数据库
        result = process_knowledge_graph(kg_data, source_document)
        
        # 打印最终结果
        logger.info("=== 知识图谱构建完成 ===")
        logger.info(f"来源文档: {result['source_document']}")
        logger.info(f"处理概念数: {result['concepts_processed']}")
        logger.info(f"处理关系数: {result['relations_processed']}")
        
        return result
        
    except Exception as e:
        logger.error(f"工作流执行失败: {str(e)}")
        return None

if __name__ == "__main__":
    # 示例用法
    # 从文件处理
    main(input_file="userinput_半结构化文本.txt")
    
    # 或直接处理文本
    # sample_text = "机器学习是人工智能的一个分支，专注于开发能够从数据中学习的系统。深度学习是机器学习的一个特殊形式，它使用多层神经网络。"
    # main(text=sample_text)