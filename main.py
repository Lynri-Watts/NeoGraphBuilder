#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PDF文献知识图谱构建主程序

该脚本实现：
1. 读取PDF文献
2. 将文献文本分成适当长度的块（带有重叠部分）
3. 为每个块调用workflow模块处理
4. 合并所有块的知识图谱结果
"""

import os
import re
import sys
import time
import json
import logging
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import PyPDF2
import textwrap

# 导入自定义模块
from database import Neo4jKnowledgeGraph
from extractor import Extractor, NEO4J_CONFIG

# 配置日志
logging.basicConfig(
    level=logging.DEBUG,  # 设置为DEBUG级别，以便查看详细调试信息
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 配置参数
CHUNK_SIZE = 5000  # 每个块的字符数
OVERLAP_SIZE = 500  # 重叠部分的字符数
MAX_RETRIES = 3  # 处理失败时的重试次数

def read_pdf(pdf_path: str) -> str:
    """
    从PDF文件中提取文本
    
    Args:
        pdf_path: PDF文件路径
        
    Returns:
        str: 提取的文本内容
        
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件为空或无效
        Exception: 其他读取错误
    """
    # 首先检查文件是否存在
    if not os.path.exists(pdf_path):
        logger.error(f"PDF文件不存在: {pdf_path}")
        raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
    
    # 检查文件大小
    file_size = os.path.getsize(pdf_path)
    if file_size == 0:
        logger.error(f"PDF文件为空: {pdf_path}")
        raise ValueError(f"PDF文件为空: {pdf_path}")
    
    logger.info(f"开始处理PDF文件: {pdf_path}, 文件大小: {file_size} 字节")
    
    try:
        with open(pdf_path, 'rb') as file:
            # 尝试创建PDF阅读器对象
            try:
                reader = PyPDF2.PdfReader(file)
            except PyPDF2.errors.PdfReadError as e:
                logger.error(f"无效的PDF文件格式: {pdf_path}, 错误: {str(e)}")
                raise ValueError(f"无效的PDF文件: {pdf_path}, 错误: {str(e)}") from e
            
            # 检查页数
            num_pages = len(reader.pages)
            logger.info(f"PDF文件包含 {num_pages} 页")
            
            # 提取文本
            text = ""
            for page_num in range(num_pages):
                try:
                    page = reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n\n"
                    logger.debug(f"成功提取第 {page_num+1}/{num_pages} 页文本")
                except Exception as page_error:
                    logger.warning(f"提取第 {page_num+1} 页失败: {str(page_error)}")
                    # 继续处理下一页而不是整个文件失败
            
            if not text.strip():
                logger.warning(f"警告: PDF文件 '{pdf_path}' 中未提取到任何文本内容")
            else:
                logger.info(f"成功从PDF文件 '{pdf_path}' 提取文本，共 {len(text)} 字符")
            
            return text
    except (FileNotFoundError, ValueError):
        # 重新抛出这些已经处理过的特定异常
        raise
    except Exception as e:
        logger.error(f"读取PDF文件 '{pdf_path}' 时发生未预期错误: {str(e)}")
        raise Exception(f"处理PDF文件失败: {str(e)}") from e

def read_caj(caj_path: str) -> str:
    """
    从CAJ文件中提取文本
    
    Args:
        caj_path: CAJ文件路径
        
    Returns:
        str: 提取的文本内容
    """
    try:
        # 首先尝试使用caj2pdf库（这是更可靠的CAJ处理方法）
        try:
            import subprocess
            import tempfile
            import os
            
            logger.info(f"尝试使用caj2pdf工具处理CAJ文件: {caj_path}")
            
            # 使用caj2pdf命令行工具转换为临时PDF文件
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_pdf:
                temp_pdf_path = temp_pdf.name
            
            # 执行转换命令
            result = subprocess.run(
                ['caj2pdf', 'convert', '-o', temp_pdf_path, caj_path],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.warning(f"caj2pdf转换失败: {result.stderr}")
                # 清理临时文件
                if os.path.exists(temp_pdf_path):
                    os.unlink(temp_pdf_path)
                raise Exception(f"caj2pdf转换失败: {result.stderr}")
            
            # 读取转换后的PDF文件
            text = read_pdf(temp_pdf_path)
            
            # 删除临时PDF文件
            os.unlink(temp_pdf_path)
            
            logger.info(f"成功使用caj2pdf从CAJ文件 '{caj_path}' 提取文本，共 {len(text)} 字符")
            return text
        except (ImportError, FileNotFoundError):
            logger.info("caj2pdf工具未找到，尝试使用pymupdf作为备选方法")
            # 尝试使用python-mupdf/pymupdf库读取CAJ文件（仅部分CAJ格式可能支持）
            try:
                import fitz  # pymupdf
                
                logger.info(f"尝试使用pymupdf处理CAJ文件: {caj_path}")
                try:
                    doc = fitz.open(caj_path)
                    text = ""
                    
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        page_text = page.get_text()
                        if page_text:
                            text += page_text + "\n\n"
                    
                    doc.close()
                    logger.info(f"成功从CAJ文件 '{caj_path}' 提取文本，共 {len(text)} 字符")
                    return text
                except Exception as fitz_error:
                    # pymupdf可能不支持某些CAJ格式
                    logger.error(f"pymupdf无法处理CAJ文件: {str(fitz_error)}")
                    raise ImportError(
                        "CAJ文件处理失败！\n" +
                        "请安装caj2pdf工具以支持CAJ格式: \n" +
                        "1. 访问 https://github.com/caj2pdf/caj2pdf 下载并安装caj2pdf\n" +
                        "2. 确保caj2pdf命令可以在命令行中运行\n" +
                        "3. 或者将CAJ文件手动转换为PDF格式后再使用本程序处理"
                    )
            except ImportError:
                raise ImportError(
                    "CAJ文件处理失败！\n" +
                    "请安装所需的库以支持CAJ文件格式: \n" +
                    "1. 安装pymupdf: pip install pymupdf\n" +
                    "2. 安装caj2pdf工具: 访问 https://github.com/caj2pdf/caj2pdf\n" +
                    "或者将CAJ文件手动转换为PDF格式后再使用本程序处理"
                )
        except Exception as e:
            logger.error(f"CAJ文件处理失败: {str(e)}")
            raise Exception(f"CAJ文件处理失败: {str(e)}")
    except Exception as e:
        logger.error(f"读取CAJ文件失败: {str(e)}")
        raise

def read_document(file_path: str) -> str:
    """
    根据文件扩展名读取文档（目前仅支持PDF格式）
    
    Args:
        file_path: 文件路径
        
    Returns:
        str: 提取的文本内容
        
    Raises:
        FileNotFoundError: 如果文件不存在
        ValueError: 如果文件格式不支持或文件无效
        Exception: 其他读取错误
    """
    logger.info(f"开始处理文档: {file_path}")
    
    # 首先检查文件是否存在
    if not os.path.exists(file_path):
        logger.error(f"文件不存在: {file_path}")
        raise FileNotFoundError(f"文件不存在: {file_path}")
    
    # 获取文件扩展名
    file_ext = os.path.splitext(file_path)[1].lower()
    
    # 添加详细日志记录，帮助诊断文件类型判断问题
    logger.debug(f"检测到的扩展名: '{file_ext}'")
    logger.debug(f"扩展名长度: {len(file_ext)}")
    logger.debug(f"扩展名十六进制表示: {[hex(ord(c)) for c in file_ext]}")
    
    # 使用strip()移除可能的空白字符
    clean_ext = file_ext.strip()
    logger.debug(f"清理后的扩展名: '{clean_ext}'")
    
    # 根据扩展名调用不同的读取函数
    if clean_ext == '.pdf':
        logger.info(f"识别为PDF文件: {file_path}")
        try:
            return read_pdf(file_path)
        except (FileNotFoundError, ValueError) as e:
            # 传递这些已知异常
            logger.error(f"PDF处理失败: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"PDF处理时发生未预期错误: {str(e)}")
            raise Exception(f"处理PDF文件失败: {str(e)}") from e
    else:
        logger.error(f"不支持的文件类型: '{clean_ext}'。目前仅支持PDF格式。")
        raise ValueError(f"不支持的文件格式: {clean_ext}。目前仅支持PDF格式。")

def split_text_with_overlap(text: str, chunk_size: int, overlap_size: int) -> List[Tuple[str, int, int]]:
    """
    将文本分成带有重叠部分的块
    
    Args:
        text: 输入文本
        chunk_size: 每个块的大小（字符数）
        overlap_size: 块之间的重叠大小（字符数）
        
    Returns:
        List[Tuple[str, int, int]]: 包含(块文本, 起始位置, 结束位置)的列表
    """
    if chunk_size <= overlap_size:
        raise ValueError("块大小必须大于重叠大小")
    
    chunks = []
    start = 0
    text_length = len(text)
    
    # 计算步长（块大小减去重叠大小）
    step = chunk_size - overlap_size
    
    while start < text_length:
        # 计算当前块的结束位置
        end = min(start + chunk_size, text_length)
        
        # 提取块文本
        chunk_text = text[start:end]
        chunks.append((chunk_text, start, end))
        
        # 更新起始位置
        start += step
    
    logger.info(f"文本已分成 {len(chunks)} 个块，每个块约 {chunk_size} 字符，重叠 {overlap_size} 字符")
    return chunks

def process_document_to_knowledge_graph(file_path: str, chunk_size: int = CHUNK_SIZE, overlap_size: int = OVERLAP_SIZE, disable_report: bool = True, disable_chunking: bool = True) -> dict:
    """
    处理文献（仅支持PDF）并构建知识图谱
    
    Args:
        file_path: 文件路径（仅支持PDF格式）
        chunk_size: 每个块的大小（字符数）
        overlap_size: 块之间的重叠大小（字符数）
        disable_report: 是否禁用报告生成
        
    Returns:
        dict: 处理结果汇总
        
    Raises:
        FileNotFoundError: 文件不存在时抛出
        ValueError: 文件格式不支持时抛出
        Exception: 处理失败时抛出异常
    """
    # NEO4J_CONFIG已经从extractor模块直接导入
    
    # 初始化Neo4j知识图谱连接 - 在函数开始时创建一个实例，确保整个处理过程使用同一实例
    kg = Neo4jKnowledgeGraph(
        uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    
    # 验证文件存在
    if not os.path.exists(file_path):
        error_msg = f"文件不存在: {file_path}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    try:
        # 读取文档内容
        logger.info(f"开始处理文件: {file_path}")
        text = read_document(file_path)
        
        # 汇总统计
        total_concepts = 0
        total_relations = 0
        processed_chunks = 0
        failed_chunks = 0
        
        # 用于记录每个块的新增节点和关系数
        chunk_stats = []
        
        # 第一步：对整篇文章进行完整提取
        logger.info("=== 第一步：对整篇文章进行完整提取 ===")
        source_document = f"{os.path.basename(file_path)}"
        
        retry_count = 0
        success = False
        full_document_result = None
        last_error = None
        
        while retry_count < MAX_RETRIES and not success:
            try:
                # 记录处理前的节点和关系数
                before_nodes = kg.stats['nodes']['new_concepts'] + kg.stats['nodes']['new_entities']
                before_relations = kg.stats['relations']['new_relations']
                
                # 调用Extractor类处理整篇文章，传入kg实例和source_document
                extractor = Extractor()
                full_document_result = extractor.process(text=text, kg=kg, source_document=source_document)
                
                if full_document_result:
                    # 计算新增的节点和关系数
                    after_nodes = kg.stats['nodes']['new_concepts'] + kg.stats['nodes']['new_entities']
                    after_relations = kg.stats['relations']['new_relations']
                    new_nodes = after_nodes - before_nodes
                    new_relations = after_relations - before_relations
                    
                    # 记录块统计信息
                    chunk_stats.append({
                        'type': 'full_document',
                        'id': 'full',
                        'new_nodes': new_nodes,
                        'new_relations': new_relations
                    })
                    
                    # 更新统计
                    total_concepts += full_document_result.get('concepts_processed', 0)
                    total_relations += full_document_result.get('relations_processed', 0)
                    processed_chunks += 1
                    success = True
                    logger.info(f"整篇文章处理成功: {full_document_result.get('concepts_processed', 0)} 个概念, {full_document_result.get('relations_processed', 0)} 个关系, 新增节点: {new_nodes}, 新增关系: {new_relations}")
                else:
                    retry_count += 1
                    error_msg = f"整篇文章处理失败，尝试重试 {retry_count}/{MAX_RETRIES}"
                    logger.warning(error_msg)
                    last_error = error_msg
                    
            except Exception as e:
                retry_count += 1
                error_msg = f"整篇文章处理异常 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}"
                logger.error(error_msg)
                last_error = str(e)
        
        if not success:
            failed_chunks += 1
            error_msg = f"整篇文章在 {MAX_RETRIES} 次尝试后仍然处理失败"
            logger.error(error_msg)
            # 如果整篇文章处理失败，抛出异常
            raise Exception(f"{error_msg}. 最后错误: {last_error}")
        
        # 第二步：分章节/分块提取（更细粒度的处理）
        if not disable_chunking:
            logger.info("=== 第二步：分章节/分块进行细粒度提取 ===")
            chunks = split_text_with_overlap(text, chunk_size, overlap_size)
            
            # 处理每个块
            for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
                chunk_id = i + 1
                logger.info(f"处理块 {chunk_id}/{len(chunks)} (字符位置: {start_pos}-{end_pos})")
                
                # 生成块标识符
                source_document = f"{os.path.basename(file_path)}_chunk_{chunk_id}"
                
                # 尝试处理块（带重试）
                retry_count = 0
                success = False
                chunk_last_error = None
                
                while retry_count < MAX_RETRIES and not success:
                    try:
                        # 记录处理前的节点和关系数
                        before_nodes = kg.stats['nodes']['new_concepts'] + kg.stats['nodes']['new_entities']
                        before_relations = kg.stats['relations']['new_relations']
                        
                        # 调用Extractor类处理当前块，传入kg实例和source_document
                        extractor = Extractor()
                        result = extractor.process(text=chunk_text, kg=kg, source_document=source_document)
                        
                        if result:
                            # 计算新增的节点和关系数
                            after_nodes = kg.stats['nodes']['new_concepts'] + kg.stats['nodes']['new_entities']
                            after_relations = kg.stats['relations']['new_relations']
                            new_nodes = after_nodes - before_nodes
                            new_relations = after_relations - before_relations
                            
                            # 记录块统计信息
                            chunk_stats.append({
                                'type': 'chunk',
                                'id': chunk_id,
                                'new_nodes': new_nodes,
                                'new_relations': new_relations
                            })
                            
                            # 更新统计
                            total_concepts += result.get('concepts_processed', 0)
                            total_relations += result.get('relations_processed', 0)
                            processed_chunks += 1
                            success = True
                            logger.info(f"块 {chunk_id} 处理成功: {result.get('concepts_processed', 0)} 个概念, {result.get('relations_processed', 0)} 个关系, 新增节点: {new_nodes}, 新增关系: {new_relations}")
                        else:
                            retry_count += 1
                            error_msg = f"块 {chunk_id} 处理失败，尝试重试 {retry_count}/{MAX_RETRIES}"
                            logger.warning(error_msg)
                            chunk_last_error = error_msg
                            
                    except Exception as e:
                        retry_count += 1
                        error_msg = f"块 {chunk_id} 处理异常 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}"
                        logger.error(error_msg)
                        chunk_last_error = str(e)
                
                if not success:
                    failed_chunks += 1
                    error_msg = f"块 {chunk_id} 在 {MAX_RETRIES} 次尝试后仍然处理失败"
                    logger.error(error_msg)
                    # 如果任何块处理失败，抛出异常
                    raise Exception(f"{error_msg}. 最后错误: {chunk_last_error}")
        logger.warning("章节提取功能暂不可用")
        # # 尝试基于标题分割章节（如果文本中有明显的章节标记）
        # logger.info("=== 第三步：尝试基于内容特征进行章节级提取 ===")
        # chapter_attempted = False
        # try:
        #     # 简单的章节检测（基于常见的章节格式，如"1. 标题", "第一章", "Section 1"等）
        #     import re
        #     # 匹配常见的章节标记，但排除参考文献
        #     chapter_markers = [
        #         r'(?:\d+\.)+(?![^\n]*[0-9]{4})\s+[^\n]+(?![^\n]*参考文献|参考文献[^\n]*|References|REFERENCES|Bibliography|BIBLIOGRAPHY)',  # 1.1. 这样的标记，排除参考文献格式
        #         r'(?:第[一二三四五六七八九十百千]+章)\s+[^\n]+(?![^\n]*参考文献)',  # 第一章 这样的标记，排除参考文献
        #         r'(?:Section|SECTION|Chapter|CHAPTER)\s+\d+\.?\s+[^\n]+(?![^\n]*(?:References|REFERENCES|Bibliography|BIBLIOGRAPHY))'  # Section 1 这样的标记，排除参考文献
        #     ]
            
        #     # 参考文献标记关键词
        #     reference_keywords = [
        #         '参考文献', 'References', 'REFERENCES', 'Bibliography', 'BIBLIOGRAPHY',
        #         'References and Notes', 'REFERENCES AND NOTES', 'Literature Cited', 'LITERATURE CITED'
        #     ]
            
        #     chapters = []
        #     for marker in chapter_markers:
        #         matches = list(re.finditer(marker, text))
        #         if matches:
        #             # 提取章节内容，但排除参考文献部分
        #             filtered_matches = []
        #             for match in matches:
        #                 title = match.group()
        #                 # 检查标题是否包含参考文献相关关键词
        #                 is_reference = False
        #                 for keyword in reference_keywords:
        #                     if keyword.lower() in title.lower():
        #                         is_reference = True
        #                         break
        #                 if not is_reference:
        #                     filtered_matches.append(match)
                    
        #             # 提取过滤后的章节
        #             if filtered_matches:
        #                 for i in range(len(filtered_matches)):
        #                     start = filtered_matches[i].start()
        #                     if i < len(filtered_matches) - 1:
        #                         end = filtered_matches[i + 1].start()
        #                     else:
        #                         # 找到最后一个章节结束位置或参考文献开始位置
        #                         last_end = len(text)
        #                         for keyword in reference_keywords:
        #                             keyword_pos = text.lower().find(keyword.lower())
        #                             if keyword_pos > start and keyword_pos < last_end:
        #                                 last_end = keyword_pos
        #                         end = last_end
        #                     chapter_text = text[start:end]
        #                     chapters.append((chapter_text, start, end, filtered_matches[i].group()))
        #                 chapter_attempted = True
        #                 break
            
        #     # 处理检测到的章节
        #     if chapters:
        #         logger.info(f"检测到 {len(chapters)} 个章节，进行章节级提取")
        #         for i, (chapter_text, start_pos, end_pos, chapter_title) in enumerate(chapters):
        #             chapter_id = i + 1
        #             logger.info(f"处理章节 {chapter_id}/{len(chapters)}: {chapter_title[:50]}... (字符位置: {start_pos}-{end_pos})")
                    
        #             # 生成章节标识符
          #             source_document = f"{os.path.basename(file_path)}_chapter_{chapter_id}"
                    
        #             # 尝试处理章节（带重试）
        #             retry_count = 0
        #             success = False
                    
        #             while retry_count < MAX_RETRIES and not success:
        #                 try:
        #                     # 调用Extractor类处理当前章节
        # extractor = Extractor()
        # result = extractor.process(text=chapter_text, kg=kg)
                            
        #                     if result:
        #                         # 更新统计
        #                         total_concepts += result.get('concepts_processed', 0)
        #                         total_relations += result.get('relations_processed', 0)
        #                         processed_chunks += 1
        #                         success = True
        #                         logger.info(f"章节 {chapter_id} 处理成功: {result.get('concepts_processed', 0)} 个概念, {result.get('relations_processed', 0)} 个关系")
        #                     else:
        #                         retry_count += 1
        #                         logger.warning(f"章节 {chapter_id} 处理失败，尝试重试 {retry_count}/{MAX_RETRIES}")
                                
        #                 except Exception as e:
        #                     retry_count += 1
        #                     logger.error(f"章节 {chapter_id} 处理异常 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}")
                    
        #             if not success:
        #                 failed_chunks += 1
        #                 logger.error(f"章节 {chapter_id} 在 {MAX_RETRIES} 次尝试后仍然处理失败")
        #     else:
        #         logger.info("未检测到明显的章节标记，跳过章节级提取")
        # except Exception as e:
        #     logger.error(f"章节提取过程中出现异常: {str(e)}")
        #     chapter_attempted = True
        
        # 生成最终汇总报告
        summary = {
            "file_path": file_path,
            "file_name": os.path.basename(file_path),
            "total_chunks": len(chunks) if 'chunks' in locals() else 0,
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "total_concepts_extracted": total_concepts,
            "total_relations_extracted": total_relations,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "extraction_method": "full_document_and_chunks",
            # "chapter_extraction_attempted": chapter_attempted
            "chunk_stats": chunk_stats  # 添加每个块的统计信息
        }
        
        # 打印最终汇总
        logger.info("=== 文献知识图谱构建完成 ===")
        logger.info(f"文件: {summary['file_path']}")
        logger.info(f"处理策略: 三级提取（全文+分块+章节）")
        logger.info(f"总处理单元数: {summary['processed_chunks']} 成功, {summary['failed_chunks']} 失败")
        logger.info(f"分块总数: {summary['total_chunks']}")
        # logger.info(f"章节提取尝试: {'是' if summary['chapter_extraction_attempted'] else '否'}")
        logger.info(f"提取概念总数: {summary['total_concepts_extracted']}")
        logger.info(f"提取关系总数: {summary['total_relations_extracted']}")
        logger.info("注：由于采用多重提取策略，部分概念和关系可能被重复提取，")
        logger.info("但数据库层面会进行去重和合并处理，确保知识图谱的一致性。")
        
        # 在整个处理流程中使用同一个kg实例
        # 如果不禁用报告，则生成详细报告和可视化
        if not disable_report:
            generate_knowledge_graph_report(kg, summary, file_path)
        
        return summary
        
    except Exception as e:
        logger.error(f"处理文献失败: {str(e)}")
        raise

def generate_knowledge_graph_report(kg: Neo4jKnowledgeGraph, summary: Dict, file_path: str):
    """
    生成知识图谱构建报告，包括统计信息和可视化
    
    Args:
        kg: Neo4jKnowledgeGraph实例，包含统计数据
        summary: 处理摘要信息
        file_path: 文件路径
    """
    logger.info("\n=== 知识图谱构建报告 ===")
    logger.info(f"文件: {os.path.basename(file_path)}")
    
    # 1. 节点统计信息
    logger.info("\n1. 节点统计:")
    logger.info(f"   新增概念数: {kg.stats['nodes']['new_concepts']}")
    logger.info(f"   合并概念数: {kg.stats['nodes']['merged_concepts']}")
    logger.info(f"   新增实体数: {kg.stats['nodes']['new_entities']}")
    logger.info(f"   合并实体数: {kg.stats['nodes']['merged_entities']}")
    total_new_nodes = kg.stats['nodes']['new_concepts'] + kg.stats['nodes']['new_entities']
    total_merged_nodes = kg.stats['nodes']['merged_concepts'] + kg.stats['nodes']['merged_entities']
    logger.info(f"   总新增节点: {total_new_nodes}")
    logger.info(f"   总合并节点: {total_merged_nodes}")
    
    # 2. 关系统计信息
    logger.info("\n2. 关系统计:")
    logger.info(f"   新增关系数: {kg.stats['relations']['new_relations']}")
    logger.info(f"   合并关系数: {kg.stats['relations']['merged_relations']}")
    
    # 3. 相似度匹配方法统计
    logger.info("\n3. 相似度判断方法贡献:")
    total_matches = sum(kg.stats['similarity_matches'].values())
    for method, count in kg.stats['similarity_matches'].items():
        percentage = (count / total_matches * 100) if total_matches > 0 else 0
        method_name = {
            'exact_match': '精确匹配',
            'alias_match': '别名匹配',
            'fuzzy_match': '模糊匹配',
            'vector_match': '词向量匹配',
            'context_match': '上下文匹配'
        }.get(method, method)
        logger.info(f"   {method_name}: {count} ({percentage:.1f}%)")
    
    # 4. 每个块的新增节点和关系数统计
    if 'chunk_stats' in summary and summary['chunk_stats']:
        logger.info("\n4. 各处理块新增节点和关系统计:")
        for chunk in summary['chunk_stats']:
            if chunk['type'] == 'full_document':
                logger.info(f"   全文: 新增节点 {chunk['new_nodes']}, 新增关系 {chunk['new_relations']}")
            else:
                logger.info(f"   块 {chunk['id']}: 新增节点 {chunk['new_nodes']}, 新增关系 {chunk['new_relations']}")
    
    # 保存统计数据到JSON文件
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_data = {
        'file': os.path.basename(file_path),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': summary,
        'stats': kg.stats
    }
    
    report_filename = f"report_{os.path.splitext(os.path.basename(file_path))[0]}_{int(time.time())}.json"
    report_path = os.path.join(report_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n报告数据已保存到: {report_path}")
    
    try:
        # 生成可视化图表
        generate_visualizations(kg.stats, summary, report_dir, os.path.basename(pdf_path))
    except Exception as e:
        logger.error(f"生成可视化时出错: {str(e)}")

def generate_visualizations(stats: Dict, summary: Dict, report_dir: str, pdf_name: str):
    """
    生成统计数据的可视化图表
    
    Args:
        stats: 统计数据字典
        summary: 处理摘要信息，包含块统计数据
        report_dir: 报告保存目录
        pdf_name: PDF文件名
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建一个包含多个子图的图表
    # 如果有块统计数据，则增加图表大小以容纳更多子图
    if 'chunk_stats' in summary and summary['chunk_stats']:
        fig = plt.figure(figsize=(25, 20))
        layout = (3, 2)
    else:
        fig = plt.figure(figsize=(20, 15))
        layout = (2, 2)
    
    # 1. 节点统计饼图
    ax1 = plt.subplot(layout[0], layout[1], 1)
    node_labels = ['新增概念', '合并概念', '新增实体', '合并实体']
    node_sizes = [
        stats['nodes']['new_concepts'],
        stats['nodes']['merged_concepts'],
        stats['nodes']['new_entities'],
        stats['nodes']['merged_entities']
    ]
    node_colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
    ax1.pie(node_sizes, labels=node_labels, colors=node_colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('节点类型分布')
    ax1.axis('equal')  # 保证饼图是圆的
    
    # 2. 相似度匹配方法柱状图
    ax2 = plt.subplot(layout[0], layout[1], 2)
    match_methods = ['精确匹配', '别名匹配', '模糊匹配', '词向量匹配', '上下文匹配']
    match_counts = [
        stats['similarity_matches']['exact_match'],
        stats['similarity_matches']['alias_match'],
        stats['similarity_matches']['fuzzy_match'],
        stats['similarity_matches']['vector_match'],
        stats['similarity_matches']['context_match']
    ]
    
    # 计算百分比
    total = sum(match_counts)
    if total > 0:
        match_percentages = [count / total * 100 for count in match_counts]
    else:
        match_percentages = [0] * len(match_counts)
    
    # 创建柱状图
    bars = ax2.bar(match_methods, match_percentages, color='skyblue')
    ax2.set_title('相似度判断方法贡献', fontsize=14)
    ax2.set_xlabel('匹配方法', fontsize=12)
    ax2.set_ylabel('贡献比例 (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    
    # 在柱状图上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. 新增vs合并对比柱状图
    ax3 = plt.subplot(layout[0], layout[1], 3)
    categories = ['概念', '实体', '关系']
    new_values = [
        stats['nodes']['new_concepts'],
        stats['nodes']['new_entities'],
        stats['relations']['new_relations']
    ]
    merged_values = [
        stats['nodes']['merged_concepts'],
        stats['nodes']['merged_entities'],
        stats['relations']['merged_relations']
    ]
    
    x = range(len(categories))
    width = 0.35
    
    ax3.bar([i - width/2 for i in x], new_values, width, label='新增', color='green')
    ax3.bar([i + width/2 for i in x], merged_values, width, label='合并', color='orange')
    
    ax3.set_title('新增vs合并对比', fontsize=14)
    ax3.set_xlabel('类型', fontsize=12)
    ax3.set_ylabel('数量', fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, fontsize=10)
    ax3.tick_params(axis='y', labelsize=10)
    ax3.legend(fontsize=12)
    
    # 4. 相似度方法热力图
    ax4 = plt.subplot(layout[0], layout[1], 4)
    # 创建一个简单的热力图数据
    heatmap_data = [match_counts]
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax4,
               xticklabels=match_methods, yticklabels=['匹配次数'], annot_kws={'size': 12})
    ax4.set_title('相似度方法使用频次热力图', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    ax4.tick_params(axis='y', labelsize=10)
    
    # 5. 每个块的新增节点和关系数条形图（如果有块统计数据）
    if 'chunk_stats' in summary and summary['chunk_stats']:
        ax5 = plt.subplot(layout[0], layout[1], 5)
        
        # 准备数据
        chunk_labels = []
        new_nodes_list = []
        new_relations_list = []
        
        for chunk in summary['chunk_stats']:
            if chunk['type'] == 'full_document':
                chunk_labels.append('全文')
            else:
                chunk_labels.append(f'块 {chunk["id"]}')
            new_nodes_list.append(chunk['new_nodes'])
            new_relations_list.append(chunk['new_relations'])
        
        # 创建条形图
        x = range(len(chunk_labels))
        width = 0.35
        
        ax5.bar([i - width/2 for i in x], new_nodes_list, width, label='新增节点', color='blue')
        ax5.bar([i + width/2 for i in x], new_relations_list, width, label='新增关系', color='red')
        
        ax5.set_title('各处理块新增节点和关系数', fontsize=14)
        ax5.set_xlabel('处理块', fontsize=12)
        ax5.set_ylabel('数量', fontsize=12)
        ax5.set_xticks(x)
        ax5.set_xticklabels(chunk_labels, rotation=45, ha='right', fontsize=10)
        ax5.legend(fontsize=12)
        ax5.tick_params(axis='y', labelsize=10)
        
        # 在柱状图上方添加数值标签
        for i, v in enumerate(new_nodes_list):
            ax5.text(i - width/2, v + 0.5, str(v), ha='center', fontsize=9)
        for i, v in enumerate(new_relations_list):
            ax5.text(i + width/2, v + 0.5, str(v), ha='center', fontsize=9)
        
        # 6. 累计节点和关系增长趋势图
        ax6 = plt.subplot(layout[0], layout[1], 6)
        
        # 计算累计值
        cumulative_nodes = []
        cumulative_relations = []
        running_nodes = 0
        running_relations = 0
        
        for chunk in summary['chunk_stats']:
            running_nodes += chunk['new_nodes']
            running_relations += chunk['new_relations']
            cumulative_nodes.append(running_nodes)
            cumulative_relations.append(running_relations)
        
        ax6.plot(range(len(chunk_labels)), cumulative_nodes, 'b-o', label='累计节点数', linewidth=2)
        ax6.plot(range(len(chunk_labels)), cumulative_relations, 'r-s', label='累计关系数', linewidth=2)
        
        ax6.set_title('累计节点和关系增长趋势', fontsize=14)
        ax6.set_xlabel('处理块', fontsize=12)
        ax6.set_ylabel('累计数量', fontsize=12)
        ax6.set_xticks(range(len(chunk_labels)))
        ax6.set_xticklabels(chunk_labels, rotation=45, ha='right', fontsize=10)
        ax6.legend(fontsize=12)
        ax6.grid(True, linestyle='--', alpha=0.7)
        ax6.tick_params(axis='y', labelsize=10)
        
        # 添加数据标签
        for i, v in enumerate(cumulative_nodes):
            ax6.text(i, v + 5, str(v), ha='center', fontsize=9, color='blue')
        for i, v in enumerate(cumulative_relations):
            ax6.text(i, v + 5, str(v), ha='center', fontsize=9, color='red')
    
    # 添加整体标题
    fig.suptitle(f'知识图谱构建报告 - {pdf_name}', fontsize=18, y=0.98)
    
    # 调整布局
    if 'chunk_stats' in summary and summary['chunk_stats']:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    else:
        plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    # 保存图表
    viz_filename = f"visualization_{os.path.splitext(pdf_name)[0]}_{int(time.time())}.png"
    viz_path = os.path.join(report_dir, viz_filename)
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"可视化图表已保存到: {viz_path}")
    logger.info("报告生成完成！请在reports文件夹中查看详细报告和可视化图表。")

def main():
    """
    主函数 - 处理命令行参数并启动处理
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='文献知识图谱构建工具（仅支持PDF格式）')
    parser.add_argument('file_path', help='文件路径（仅支持PDF格式）')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help=f'每个块的字符数 (默认: {CHUNK_SIZE})')
    parser.add_argument('--overlap-size', type=int, default=OVERLAP_SIZE, help=f'块之间的重叠字符数 (默认: {OVERLAP_SIZE})')
    parser.add_argument('--enable-report', action='store_true', help='启用报告和可视化生成（默认禁用）')
    parser.add_argument('--enable-chunking', action='store_true', help='启用分块处理（默认禁用）')
    
    args = parser.parse_args()
    
    try:
        # 执行处理
        start_time = time.time()
        logger.info(f"开始处理文件: {args.file_path}")
        
        # 根据命令行参数决定是否禁用报告和是否启用分块处理
        # 注意：disable_chunking参数与enable-chunking命令行参数相反
        process_document_to_knowledge_graph(args.file_path, args.chunk_size, args.overlap_size, not args.enable_report, not args.enable_chunking)
        
        end_time = time.time()
        logger.info(f"文件处理完成，总耗时: {(end_time - start_time):.2f} 秒")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise

# 为了保持向后兼容性，保留原函数名
process_pdf_to_knowledge_graph = process_document_to_knowledge_graph

if __name__ == "__main__":
    main()