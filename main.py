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
import logging
from typing import List, Tuple
import PyPDF2
import textwrap

# 导入workflow模块
from workflow import main as workflow_main

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
    """
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
            
            logger.info(f"成功从PDF文件 '{pdf_path}' 提取文本，共 {len(text)} 字符")
            return text
    except Exception as e:
        logger.error(f"读取PDF文件失败: {str(e)}")
        raise

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

def process_pdf_to_knowledge_graph(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap_size: int = OVERLAP_SIZE) -> dict:
    """
    处理PDF文献并构建知识图谱
    
    Args:
        pdf_path: PDF文件路径
        chunk_size: 每个块的大小（字符数）
        overlap_size: 块之间的重叠大小（字符数）
        
    Returns:
        dict: 处理结果汇总
    """
    try:
        # 验证文件存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 读取PDF内容
        logger.info(f"开始处理PDF文件: {pdf_path}")
        text = read_pdf(pdf_path)
        
        # 分割文本为块
        chunks = split_text_with_overlap(text, chunk_size, overlap_size)
        
        # 汇总统计
        total_concepts = 0
        total_relations = 0
        processed_chunks = 0
        failed_chunks = 0
        
        # 处理每个块
        for i, (chunk_text, start_pos, end_pos) in enumerate(chunks):
            chunk_id = i + 1
            logger.info(f"处理块 {chunk_id}/{len(chunks)} (字符位置: {start_pos}-{end_pos})")
            
            # 生成块标识符
            source_document = f"{os.path.basename(pdf_path)}_chunk_{chunk_id}"
            
            # 尝试处理块（带重试）
            retry_count = 0
            success = False
            
            while retry_count < MAX_RETRIES and not success:
                try:
                    # 调用workflow处理当前块
                    result = workflow_main(text=chunk_text)
                    
                    if result:
                        # 更新统计
                        total_concepts += result.get('concepts_processed', 0)
                        total_relations += result.get('relations_processed', 0)
                        processed_chunks += 1
                        success = True
                        logger.info(f"块 {chunk_id} 处理成功: {result.get('concepts_processed', 0)} 个概念, {result.get('relations_processed', 0)} 个关系")
                    else:
                        retry_count += 1
                        logger.warning(f"块 {chunk_id} 处理失败，尝试重试 {retry_count}/{MAX_RETRIES}")
                        
                except Exception as e:
                    retry_count += 1
                    logger.error(f"块 {chunk_id} 处理异常 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}")
            
            if not success:
                failed_chunks += 1
                logger.error(f"块 {chunk_id} 在 {MAX_RETRIES} 次尝试后仍然处理失败")
        
        # 生成最终汇总报告
        summary = {
            "pdf_file": pdf_path,
            "total_chunks": len(chunks),
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "total_concepts_extracted": total_concepts,
            "total_relations_extracted": total_relations,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size
        }
        
        # 打印最终汇总
        logger.info("=== PDF知识图谱构建完成 ===")
        logger.info(f"PDF文件: {summary['pdf_file']}")
        logger.info(f"处理块数: {summary['processed_chunks']}/{summary['total_chunks']}")
        logger.info(f"失败块数: {summary['failed_chunks']}")
        logger.info(f"提取概念总数: {summary['total_concepts_extracted']}")
        logger.info(f"提取关系总数: {summary['total_relations_extracted']}")
        
        return summary
        
    except Exception as e:
        logger.error(f"处理PDF文献失败: {str(e)}")
        raise

def main():
    """
    主函数 - 处理命令行参数并启动处理
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='PDF文献知识图谱构建工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help=f'每个块的字符数 (默认: {CHUNK_SIZE})')
    parser.add_argument('--overlap-size', type=int, default=OVERLAP_SIZE, help=f'块之间的重叠字符数 (默认: {OVERLAP_SIZE})')
    
    args = parser.parse_args()
    
    # 执行处理
    process_pdf_to_knowledge_graph(args.pdf_path, args.chunk_size, args.overlap_size)

if __name__ == "__main__":
    main()