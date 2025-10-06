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
import workflow
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

def process_pdf_to_knowledge_graph(pdf_path: str, chunk_size: int = CHUNK_SIZE, overlap_size: int = OVERLAP_SIZE, disable_report: bool = False) -> dict:
    """
    处理PDF文献并构建知识图谱
    
    Args:
        pdf_path: PDF文件路径
        chunk_size: 每个块的大小（字符数）
        overlap_size: 块之间的重叠大小（字符数）
        disable_report: 是否禁用报告生成
        
    Returns:
        dict: 处理结果汇总
    """
    # 从workflow模块获取Neo4j配置
    from workflow import NEO4J_CONFIG
    
    # 初始化Neo4j知识图谱连接 - 在函数开始时创建一个实例，确保整个处理过程使用同一实例
    kg = Neo4jKnowledgeGraph(
        uri=NEO4J_CONFIG["uri"],
        user=NEO4J_CONFIG["user"],
        password=NEO4J_CONFIG["password"]
    )
    try:
        # 验证文件存在
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 读取PDF内容
        logger.info(f"开始处理PDF文件: {pdf_path}")
        text = read_pdf(pdf_path)
        
        # 汇总统计
        total_concepts = 0
        total_relations = 0
        processed_chunks = 0
        failed_chunks = 0
        
        # 第一步：对整篇文章进行完整提取
        logger.info("=== 第一步：对整篇文章进行完整提取 ===")
        source_document = f"{os.path.basename(pdf_path)}_full_document"
        
        retry_count = 0
        success = False
        full_document_result = None
        
        while retry_count < MAX_RETRIES and not success:
            try:
                # 调用workflow处理整篇文章，传入kg实例
                full_document_result = workflow_main(text=text, kg=kg)
                
                if full_document_result:
                    # 更新统计
                    total_concepts += full_document_result.get('concepts_processed', 0)
                    total_relations += full_document_result.get('relations_processed', 0)
                    processed_chunks += 1
                    success = True
                    logger.info(f"整篇文章处理成功: {full_document_result.get('concepts_processed', 0)} 个概念, {full_document_result.get('relations_processed', 0)} 个关系")
                else:
                    retry_count += 1
                    logger.warning(f"整篇文章处理失败，尝试重试 {retry_count}/{MAX_RETRIES}")
                    
            except Exception as e:
                retry_count += 1
                logger.error(f"整篇文章处理异常 (尝试 {retry_count}/{MAX_RETRIES}): {str(e)}")
        
        if not success:
            failed_chunks += 1
            logger.error(f"整篇文章在 {MAX_RETRIES} 次尝试后仍然处理失败")
        
        # 第二步：分章节/分块提取（更细粒度的处理）
        logger.info("=== 第二步：分章节/分块进行细粒度提取 ===")
        chunks = split_text_with_overlap(text, chunk_size, overlap_size)
        
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
                    # 调用workflow处理当前块，传入kg实例
                    result = workflow_main(text=chunk_text, kg=kg)
                    
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
        #             source_document = f"{os.path.basename(pdf_path)}_chapter_{chapter_id}"
                    
        #             # 尝试处理章节（带重试）
        #             retry_count = 0
        #             success = False
                    
        #             while retry_count < MAX_RETRIES and not success:
        #                 try:
        #                     # 调用workflow处理当前章节
        #                     result = workflow_main(text=chapter_text, kg=kg)
                            
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
            "pdf_file": pdf_path,
            "total_chunks": len(chunks) if 'chunks' in locals() else 0,
            "processed_chunks": processed_chunks,
            "failed_chunks": failed_chunks,
            "total_concepts_extracted": total_concepts,
            "total_relations_extracted": total_relations,
            "chunk_size": chunk_size,
            "overlap_size": overlap_size,
            "extraction_method": "full_document_and_chunks",
            # "chapter_extraction_attempted": chapter_attempted
        }
        
        # 打印最终汇总
        logger.info("=== PDF知识图谱构建完成 ===")
        logger.info(f"PDF文件: {summary['pdf_file']}")
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
            generate_knowledge_graph_report(kg, summary, pdf_path)
        
        return summary
        
    except Exception as e:
        logger.error(f"处理PDF文献失败: {str(e)}")
        raise

def generate_knowledge_graph_report(kg: Neo4jKnowledgeGraph, summary: Dict, pdf_path: str):
    """
    生成知识图谱构建报告，包括统计信息和可视化
    
    Args:
        kg: Neo4jKnowledgeGraph实例，包含统计数据
        summary: 处理摘要信息
        pdf_path: PDF文件路径
    """
    logger.info("\n=== 知识图谱构建报告 ===")
    logger.info(f"PDF文件: {os.path.basename(pdf_path)}")
    
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
    
    # 保存统计数据到JSON文件
    report_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reports')
    os.makedirs(report_dir, exist_ok=True)
    
    report_data = {
        'pdf_file': os.path.basename(pdf_path),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'summary': summary,
        'stats': kg.stats
    }
    
    report_filename = f"report_{os.path.splitext(os.path.basename(pdf_path))[0]}_{int(time.time())}.json"
    report_path = os.path.join(report_dir, report_filename)
    
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"\n报告数据已保存到: {report_path}")
    
    try:
        # 生成可视化图表
        generate_visualizations(kg.stats, report_dir, os.path.basename(pdf_path))
    except Exception as e:
        logger.error(f"生成可视化时出错: {str(e)}")

def generate_visualizations(stats: Dict, report_dir: str, pdf_name: str):
    """
    生成统计数据的可视化图表
    
    Args:
        stats: 统计数据字典
        report_dir: 报告保存目录
        pdf_name: PDF文件名
    """
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 创建一个包含多个子图的图表
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 节点统计饼图
    ax1 = plt.subplot(2, 2, 1)
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
    ax2 = plt.subplot(2, 2, 2)
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
    ax2.set_title('相似度判断方法贡献')
    ax2.set_xlabel('匹配方法')
    ax2.set_ylabel('贡献比例 (%)')
    ax2.set_ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    
    # 在柱状图上方添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom')
    
    # 3. 新增vs合并对比柱状图
    ax3 = plt.subplot(2, 2, 3)
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
    
    ax3.set_title('新增vs合并对比')
    ax3.set_xlabel('类型')
    ax3.set_ylabel('数量')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    
    # 4. 相似度方法热力图
    ax4 = plt.subplot(2, 2, 4)
    # 创建一个简单的热力图数据
    heatmap_data = [match_counts]
    sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu', ax=ax4,
               xticklabels=match_methods, yticklabels=['匹配次数'])
    ax4.set_title('相似度方法使用频次热力图')
    plt.xticks(rotation=45, ha='right')
    
    # 添加整体标题
    fig.suptitle(f'知识图谱构建报告 - {pdf_name}', fontsize=16, y=0.98)
    
    # 调整布局
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
    
    parser = argparse.ArgumentParser(description='PDF文献知识图谱构建工具')
    parser.add_argument('pdf_path', help='PDF文件路径')
    parser.add_argument('--chunk-size', type=int, default=CHUNK_SIZE, help=f'每个块的字符数 (默认: {CHUNK_SIZE})')
    parser.add_argument('--overlap-size', type=int, default=OVERLAP_SIZE, help=f'块之间的重叠字符数 (默认: {OVERLAP_SIZE})')
    parser.add_argument('--disable-report', action='store_true', help='禁用报告和可视化生成')
    
    args = parser.parse_args()
    
    try:
        # 执行处理
        start_time = time.time()
        logger.info(f"开始处理PDF文件: {args.pdf_path}")
        
        # 根据命令行参数决定是否禁用报告
        process_pdf_to_knowledge_graph(args.pdf_path, args.chunk_size, args.overlap_size, args.disable_report)
        
        end_time = time.time()
        logger.info(f"PDF处理完成，总耗时: {(end_time - start_time):.2f} 秒")
        
    except KeyboardInterrupt:
        logger.info("用户中断操作")
    except Exception as e:
        logger.error(f"处理过程中发生错误: {str(e)}")
        raise

if __name__ == "__main__":
    main()