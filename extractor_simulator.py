#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
知识图谱提取器模拟器
用于向数据库中插入预先设计好的测试内容，以测试database.py的功能
"""

import logging
import json
import uuid
from database import Neo4jKnowledgeGraph
import configparser
import os# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_test_concepts():
    """创建测试用的概念数据"""
    concepts = [
        {
            "name": "人工智能",
            "type": "Concept",
            "description": "人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。",
            "source_document": {
                "filename": "ai_textbook.txt",
                "title": "人工智能基础教材",
                "paragraph_number": 1
            },
            "aliases": ["AI", "机器智能", "Artificial Intelligence"]
        },
        {
            "name": "机器学习",
            "type": "Concept", 
            "description": "机器学习是人工智能的一个子领域，专注于开发能够从数据中学习的算法。",
            "source_document": {
                "filename": "ml_guide.pdf",
                "title": "机器学习指南",
                "paragraph_number": 3
            },
            "aliases": ["Machine Learning", "ML"]
        },
        {
            "name": "深度学习",
            "type": "Concept",
            "description": "深度学习是机器学习的一个子集，使用多层神经网络来学习数据的复杂模式。",
            "source_document": {
                "filename": "deep_learning_book.pdf", 
                "title": "深度学习实战",
                "paragraph_number": 5
            },
            "aliases": ["Deep Learning", "深度神经网络"]
        },
        {
            "name": "自然语言处理",
            "type": "Concept",
            "description": "自然语言处理是人工智能的一个分支，专注于计算机与人类语言之间的交互。",
            "source_document": {
                "filename": "nlp_basics.txt",
                "title": "自然语言处理基础",
                "paragraph_number": 2
            },
            "aliases": ["NLP", "Natural Language Processing"]
        },
        {
            "name": "计算机视觉",
            "type": "Concept",
            "description": "计算机视觉是人工智能的一个领域，专注于让计算机能够理解和解释视觉信息。",
            "source_document": {
                "filename": "cv_introduction.pdf",
                "title": "计算机视觉导论", 
                "paragraph_number": 4
            },
            "aliases": ["Computer Vision", "CV"]
        }
    ]
    return concepts

def create_test_entities():
    """创建测试用的实体数据"""
    entities = [
        {
            "name": "OpenAI",
            "type": "Entity",
            "description": "OpenAI是一家专注于人工智能研究和开发的公司。",
            "source_document": {
                "filename": "tech_companies.txt",
                "title": "科技公司介绍",
                "paragraph_number": 1
            },
            "aliases": ["OpenAI Inc.", "OpenAI LP"]
        },
        {
            "name": "TensorFlow",
            "type": "Entity",
            "description": "TensorFlow是Google开发的开源机器学习框架。",
            "source_document": {
                "filename": "ml_frameworks.pdf",
                "title": "机器学习框架对比",
                "paragraph_number": 2
            },
            "aliases": ["TF", "Google TensorFlow"]
        },
        {
            "name": "PyTorch",
            "type": "Entity", 
            "description": "PyTorch是Facebook开发的深度学习框架。",
            "source_document": {
                "filename": "ml_frameworks.pdf",
                "title": "机器学习框架对比",
                "paragraph_number": 3
            },
            "aliases": ["Torch", "Facebook PyTorch"]
        },
        {
            "name": "GPT-4",
            "type": "Entity",
            "description": "GPT-4是OpenAI开发的大型语言模型。",
            "source_document": {
                "filename": "llm_models.txt",
                "title": "大型语言模型概览",
                "paragraph_number": 1
            },
            "aliases": ["GPT4", "Generative Pre-trained Transformer 4"]
        },
        {
            "name": "BERT",
            "type": "Entity",
            "description": "BERT是Google开发的预训练语言模型。",
            "source_document": {
                "filename": "nlp_models.pdf",
                "title": "自然语言处理模型",
                "paragraph_number": 2
            },
            "aliases": ["Bidirectional Encoder Representations from Transformers"]
        }
    ]
    return entities

def create_test_relations():
    """创建测试用的关系数据"""
    relations = [
        {
            "source": "机器学习",
            "target": "人工智能",
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ai_hierarchy.txt",
                "title": "人工智能领域层次结构",
                "paragraph_number": 1
            }
        },
        {
            "source": "深度学习",
            "target": "机器学习", 
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ai_hierarchy.txt",
                "title": "人工智能领域层次结构",
                "paragraph_number": 2
            }
        },
        {
            "source": "自然语言处理",
            "target": "人工智能",
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ai_hierarchy.txt",
                "title": "人工智能领域层次结构",
                "paragraph_number": 3
            }
        },
        {
            "source": "计算机视觉",
            "target": "人工智能",
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ai_hierarchy.txt",
                "title": "人工智能领域层次结构",
                "paragraph_number": 4
            }
        },
        {
            "source": "OpenAI",
            "target": "GPT-4",
            "type": "DEVELOPED",
            "source_document": {
                "filename": "company_products.txt",
                "title": "公司产品关系",
                "paragraph_number": 1
            }
        },
        {
            "source": "TensorFlow",
            "target": "深度学习",
            "type": "USED_IN",
            "source_document": {
                "filename": "framework_usage.txt",
                "title": "框架应用场景",
                "paragraph_number": 1
            }
        },
        {
            "source": "PyTorch",
            "target": "深度学习",
            "type": "USED_IN",
            "source_document": {
                "filename": "framework_usage.txt",
                "title": "框架应用场景",
                "paragraph_number": 2
            }
        },
        {
            "source": "GPT-4",
            "target": "自然语言处理",
            "type": "APPLIED_IN",
            "source_document": {
                "filename": "model_applications.txt",
                "title": "模型应用领域",
                "paragraph_number": 1
            }
        },
        {
            "source": "BERT",
            "target": "自然语言处理",
            "type": "APPLIED_IN",
            "source_document": {
                "filename": "model_applications.txt",
                "title": "模型应用领域",
                "paragraph_number": 2
            }
        },
        {
            "source": "TensorFlow",
            "target": "Google",
            "type": "DEVELOPED_BY",
            "source_document": {
                "filename": "company_products.txt",
                "title": "公司产品关系",
                "paragraph_number": 2
            }
        }
    ]
    return relations

def test_duplicate_scenarios():
    """创建重复数据测试场景，测试相同名称但不同属性的节点合并逻辑"""
    # 测试相同概念但不同来源文档、描述和别名的情况
    duplicate_concepts = [
        {
            "name": "机器学习",
            "type": "Concept",
            "description": "机器学习使计算机能够从经验中学习和改进，而无需明确编程。这是第二个来源的描述。",
            "source_document": {
                "filename": "another_ml_book.pdf",
                "title": "机器学习进阶",
                "paragraph_number": 1
            },
            "aliases": ["ML", "Machine Learning", "统计学习"]  # 包含新的别名
        },
        {
            "name": "机器学习",  # 完全相同的名称，第三个来源
            "type": "Concept", 
            "description": "机器学习是一种数据分析方法，通过算法自动分析数据以发现模式。这是第三个来源的描述。",
            "source_document": {
                "filename": "ml_research_paper.pdf",
                "title": "机器学习研究论文",
                "paragraph_number": 5
            },
            "aliases": ["Machine Learning", "自动学习", "算法学习"]  # 又一组不同的别名
        },
        {
            "name": "深度学习", 
            "type": "Concept",
            "description": "深度学习使用人工神经网络来模拟人脑的学习过程。这是第二个来源的描述。",
            "source_document": {
                "filename": "neural_networks.txt",
                "title": "神经网络基础",
                "paragraph_number": 3
            },
            "aliases": ["Deep Learning", "深度神经网络", "DNN"]  # 包含新的别名
        },
        {
            "name": "深度学习",  # 完全相同的名称，第三个来源
            "type": "Concept",
            "description": "深度学习是机器学习的一个分支，使用多层结构来学习数据的表示。这是第三个来源的描述。",
            "source_document": {
                "filename": "dl_applications.pdf",
                "title": "深度学习应用",
                "paragraph_number": 2
            },
            "aliases": ["Deep Learning", "多层神经网络", "表示学习"]  # 又一组不同的别名
        },
        {
            "name": "人工智能",  # 测试主概念的重复
            "type": "Concept",
            "description": "人工智能是指由机器展现的智能，与人类和动物展示的自然智能形成对比。这是第二个来源的描述。",
            "source_document": {
                "filename": "ai_philosophy.txt",
                "title": "人工智能哲学",
                "paragraph_number": 1
            },
            "aliases": ["AI", "机器智能", "人工智慧", "Artificial Intelligence"]  # 包含新的别名
        }
    ]
    
    # 测试相同实体但不同来源文档、描述和别名的情况
    duplicate_entities = [
        {
            "name": "OpenAI",
            "type": "Entity",
            "description": "OpenAI是一家人工智能研究和部署公司，致力于确保通用人工智能造福全人类。这是第二个来源的描述。",
            "source_document": {
                "filename": "company_profile.pdf",
                "title": "OpenAI公司简介",
                "paragraph_number": 2
            },
            "aliases": ["OpenAI Inc.", "OpenAI LP", "OpenAI Lab"]  # 包含新的别名
        },
        {
            "name": "OpenAI",  # 完全相同的名称，第三个来源
            "type": "Entity",
            "description": "OpenAI成立于2015年，总部位于旧金山，是AI领域的重要参与者。这是第三个来源的描述。",
            "source_document": {
                "filename": "tech_companies_2023.txt",
                "title": "2023年科技公司概览",
                "paragraph_number": 8
            },
            "aliases": ["OpenAI Inc.", "OpenAI Corporation", "ChatGPT公司"]  # 又一组不同的别名
        },
        {
            "name": "TensorFlow",
            "type": "Entity",
            "description": "TensorFlow是Google Brain团队开发的端到端开源机器学习平台。这是第二个来源的描述。",
            "source_document": {
                "filename": "google_tools.pdf",
                "title": "Google工具集",
                "paragraph_number": 4
            },
            "aliases": ["TF", "Google TensorFlow", "TensorFlow 2.x"]  # 包含新的别名
        }
    ]
    
    # 测试相同关系但不同来源文档的情况
    duplicate_relations = [
        {
            "source": "机器学习",
            "target": "人工智能",
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ml_relationships.pdf",
                "title": "机器学习关系图",
                "paragraph_number": 1
            }
        },
        {
            "source": "机器学习",
            "target": "人工智能", 
            "type": "IS_SUBFIELD_OF",
            "source_document": {
                "filename": "ai_taxonomy.txt",
                "title": "人工智能分类体系",
                "paragraph_number": 3
            }
        },
        {
            "source": "深度学习",
            "target": "机器学习",
            "type": "IS_SUBFIELD_OF", 
            "source_document": {
                "filename": "dl_hierarchy.pdf",
                "title": "深度学习层次结构",
                "paragraph_number": 2
            }
        }
    ]
    
    return duplicate_concepts, duplicate_entities, duplicate_relations

def load_config():
    """从config.ini文件加载配置"""
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(__file__), 'config.ini')
    
    if not os.path.exists(config_path):
        logger.error(f"配置文件不存在: {config_path}")
        logger.error("请复制config.ini.example为config.ini并填入正确的配置信息")
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config.read(config_path, encoding='utf-8')
    return config

def main():
    """主函数：执行测试数据插入"""
    logger.info("开始插入测试数据到知识图谱...")
    
    try:
        # 加载配置
        config = load_config()
        
        # 初始化数据库连接（从配置文件读取连接参数）
        kg = Neo4jKnowledgeGraph(
            uri=config.get('NEO4J_CONFIG', 'uri'),
            user=config.get('NEO4J_CONFIG', 'user'), 
            password=config.get('NEO4J_CONFIG', 'password')
        )
        logger.info("成功连接到Neo4j数据库")
        
        # 1. 插入测试概念
        logger.info("=== 插入测试概念 ===")
        concepts = create_test_concepts()
        concept_count = kg.insert_nodes(concepts)
        logger.info(f"成功插入 {concept_count} 个概念")
        
        # 2. 插入测试实体
        logger.info("=== 插入测试实体 ===")
        entities = create_test_entities()
        entity_count = kg.insert_nodes(entities)
        logger.info(f"成功插入 {entity_count} 个实体")
        
        # 3. 插入测试关系
        logger.info("=== 插入测试关系 ===")
        relations = create_test_relations()
        relation_count = kg.insert_relations(relations)
        logger.info(f"成功插入 {relation_count} 个关系")
        
        # 4. 测试重复数据处理
        logger.info("=== 测试重复数据处理 ===")
        duplicate_concepts, duplicate_entities, duplicate_relations = test_duplicate_scenarios()
        
        # 插入重复概念（应该触发合并逻辑）
        dup_concept_count = kg.insert_nodes(duplicate_concepts)
        logger.info(f"处理重复概念: {dup_concept_count} 个")
        
        # 插入重复实体（应该触发合并逻辑）
        dup_entity_count = kg.insert_nodes(duplicate_entities)
        logger.info(f"处理重复实体: {dup_entity_count} 个")
        
        # 插入重复关系（应该触发合并逻辑）
        dup_relation_count = kg.insert_relations(duplicate_relations)
        logger.info(f"处理重复关系: {dup_relation_count} 个")
        
        # 5. 获取统计信息
        logger.info("=== 知识图谱统计信息 ===")
        stats = kg.get_node_statistics()
        logger.info(f"节点统计: {stats}")
        
        # 6. 获取详细统计
        detailed_stats = kg.get_node_statistics()
        logger.info(f"详细统计: {detailed_stats}")
        
        logger.info("=== 测试数据插入完成 ===")
        
    except Exception as e:
        logger.error(f"插入测试数据时出错: {str(e)}")
        import traceback
        logger.error(f"详细错误信息:\n{traceback.format_exc()}")
        
    finally:
        # 关闭数据库连接
        if 'kg' in locals():
            kg.close()
            logger.info("数据库连接已关闭")

if __name__ == "__main__":
    main()