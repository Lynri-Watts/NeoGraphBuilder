#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
批量文档导入工具

该脚本实现：
1. 遍历指定目录下的所有PDF文件
2. 使用多进程并行处理文档
3. 调用main.py将每个文档导入知识图谱
"""

import os
import sys
import argparse
import subprocess
import multiprocessing
import time
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("batch_import.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 支持的文件扩展名
SUPPORTED_EXTENSIONS = ['.pdf']  # 已移除.caj，只处理PDF文件

def is_supported_file(file_path: str) -> bool:
    """检查文件是否为支持的格式"""
    ext = os.path.splitext(file_path)[1].lower()
    return ext in SUPPORTED_EXTENSIONS

def collect_files(directory: str) -> List[str]:
    """收集目录下所有支持的文件，跳过以点开头的隐藏文件"""
    files = []
    for root, _, filenames in os.walk(directory):
        for filename in filenames:
            # 跳过以点开头的隐藏文件（如Mac系统的.DS_Store）
            if filename.startswith('.'):
                logger.debug(f"跳过隐藏文件: {filename}")
                continue
            
            file_path = os.path.join(root, filename)
            if is_supported_file(file_path):
                files.append(os.path.abspath(file_path))
    return files

def process_file(file_path: str) -> Tuple[str, bool, str]:
    """
    处理单个文件
    
    Args:
        file_path: 文件路径
        
    Returns:
        Tuple[str, bool, str]: (文件路径, 是否成功, 错误信息或空字符串)
    """
    try:
        logger.info(f"开始处理文件: {file_path}")
        start_time = time.time()
        
        # 构建命令行参数
        cmd = [
            sys.executable,  # 使用与当前脚本相同的Python解释器
            os.path.join(os.path.dirname(__file__), "main.py"),
            file_path
        ]
        
        # 执行命令
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        elapsed_time = time.time() - start_time
        logger.info(f"文件处理完成: {file_path}, 耗时: {elapsed_time:.2f}秒")
        return (file_path, True, "")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"文件处理失败: {file_path}, 错误码: {e.returncode}")
        logger.error(f"错误输出: {e.stderr}")
        return (file_path, False, f"错误码: {e.returncode}, 错误: {e.stderr}")
    except Exception as e:
        logger.error(f"处理文件时发生异常: {file_path}, 异常: {str(e)}")
        return (file_path, False, f"异常: {str(e)}")

def batch_process(directory: str, max_workers: int = 3) -> None:
    """
    批量处理目录下的所有支持的文件
    
    Args:
        directory: 要处理的目录路径
        max_workers: 最大工作进程数
    """
    # 验证目录是否存在
    if not os.path.exists(directory):
        logger.error(f"目录不存在: {directory}")
        sys.exit(1)
    
    if not os.path.isdir(directory):
        logger.error(f"指定的路径不是目录: {directory}")
        sys.exit(1)
    
    # 收集所有支持的文件
    files = collect_files(directory)
    
    if not files:
        logger.info(f"目录中没有找到支持的文件（{', '.join(SUPPORTED_EXTENSIONS)}）: {directory}")
        return
    
    logger.info(f"找到 {len(files)} 个支持的文件，将使用最多 {max_workers} 个进程并行处理")
    
    # 统计信息
    total_files = len(files)
    success_count = 0
    failure_count = 0
    failed_files = []
    
    start_time = time.time()
    
    # 使用进程池并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_file = {executor.submit(process_file, file_path): file_path for file_path in files}
        
        # 等待任务完成并处理结果
        for i, future in enumerate(as_completed(future_to_file), 1):
            file_path = future_to_file[future]
            try:
                file_path, success, error_msg = future.result()
                if success:
                    success_count += 1
                else:
                    failure_count += 1
                    failed_files.append((file_path, error_msg))
                
                # 输出进度
                logger.info(f"进度: {i}/{total_files}, 成功: {success_count}, 失败: {failure_count}")
                
            except Exception as e:
                logger.error(f"获取任务结果时发生异常: {file_path}, 异常: {str(e)}")
                failure_count += 1
                failed_files.append((file_path, f"任务异常: {str(e)}"))
    
    total_time = time.time() - start_time
    
    # 输出总结
    logger.info("=== 批量处理完成 ===")
    logger.info(f"总文件数: {total_files}")
    logger.info(f"成功处理: {success_count}")
    logger.info(f"处理失败: {failure_count}")
    logger.info(f"总耗时: {total_time:.2f}秒")
    
    if failed_files:
        logger.info("\n失败的文件列表:")
        for file_path, error_msg in failed_files:
            logger.info(f"- {file_path}: {error_msg}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量导入文档到知识图谱')
    parser.add_argument('directory', help='包含文档的目录路径')
    parser.add_argument('--workers', type=int, default=3, help='并行处理的最大进程数（默认: 3）')
    
    args = parser.parse_args()
    
    # 检查系统CPU核心数，避免设置过多进程
    cpu_count = multiprocessing.cpu_count()
    if args.workers > cpu_count:
        logger.warning(f"设置的进程数({args.workers})超过了系统CPU核心数({cpu_count})，将使用CPU核心数作为最大进程数")
        args.workers = cpu_count
    
    batch_process(args.directory, args.workers)

if __name__ == "__main__":
    main()

# python batch_import.py "documents\论文 2.0" --workers 5