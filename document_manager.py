#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档管理模块

该模块负责：
1. 读取PDF/CAJ等格式的文献
2. 提取文献内容
3. 处理文献元数据
4. 提供文献概括功能
"""

import os
import re
import logging
import PyPDF2
from typing import Dict, List, Optional, Any

# 配置日志
logger = logging.getLogger(__name__)

class DocumentManager:
    """文档管理器，负责处理文献的读取和概括"""
    
    def __init__(self):
        """初始化文档管理器"""
        self.logger = logging.getLogger(__name__)
    
    def read_pdf(self, pdf_path: str) -> str:
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
            self.logger.error(f"PDF文件不存在: {pdf_path}")
            raise FileNotFoundError(f"PDF文件不存在: {pdf_path}")
        
        # 检查文件大小
        file_size = os.path.getsize(pdf_path)
        if file_size == 0:
            self.logger.error(f"PDF文件为空: {pdf_path}")
            raise ValueError(f"PDF文件为空: {pdf_path}")
        
        self.logger.info(f"开始处理PDF文件: {pdf_path}, 文件大小: {file_size} 字节")
        
        try:
            with open(pdf_path, 'rb') as file:
                # 尝试创建PDF阅读器对象
                try:
                    reader = PyPDF2.PdfReader(file)
                except PyPDF2.errors.PdfReadError as e:
                    self.logger.error(f"无效的PDF文件格式: {pdf_path}, 错误: {str(e)}")
                    raise ValueError(f"无效的PDF文件: {pdf_path}, 错误: {str(e)}") from e
                
                # 检查页数
                num_pages = len(reader.pages)
                self.logger.info(f"PDF文件包含 {num_pages} 页")
                
                # 提取文本
                text = ""
                for page_num in range(num_pages):
                    try:
                        page = reader.pages[page_num]
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n\n"
                        self.logger.debug(f"成功提取第 {page_num+1}/{num_pages} 页文本")
                    except Exception as page_error:
                        self.logger.warning(f"提取第 {page_num+1} 页失败: {str(page_error)}")
                        # 继续处理下一页而不是整个文件失败
                
                if not text.strip():
                    self.logger.warning(f"警告: PDF文件 '{pdf_path}' 中未提取到任何文本内容")
                else:
                    self.logger.info(f"成功从PDF文件 '{pdf_path}' 提取文本，共 {len(text)} 字符")
                
                return text
        except (FileNotFoundError, ValueError):
            # 重新抛出这些已经处理过的特定异常
            raise
        except Exception as e:
            self.logger.error(f"读取PDF文件 '{pdf_path}' 时发生未预期错误: {str(e)}")
            raise Exception(f"处理PDF文件失败: {str(e)}") from e
    
    def read_caj(self, caj_path: str) -> str:
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
                
                self.logger.info(f"尝试使用caj2pdf工具处理CAJ文件: {caj_path}")
                
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
                    self.logger.warning(f"caj2pdf转换失败: {result.stderr}")
                    # 清理临时文件
                    if os.path.exists(temp_pdf_path):
                        os.unlink(temp_pdf_path)
                    raise Exception(f"caj2pdf转换失败: {result.stderr}")
                
                # 读取转换后的PDF文件
                text = self.read_pdf(temp_pdf_path)
                
                # 删除临时PDF文件
                os.unlink(temp_pdf_path)
                
                self.logger.info(f"成功使用caj2pdf从CAJ文件 '{caj_path}' 提取文本，共 {len(text)} 字符")
                return text
            except (ImportError, FileNotFoundError):
                self.logger.info("caj2pdf工具未找到，尝试使用pymupdf作为备选方法")
                # 尝试使用python-mupdf/pymupdf库读取CAJ文件（仅部分CAJ格式可能支持）
                try:
                    import fitz  # pymupdf
                    
                    self.logger.info(f"尝试使用pymupdf处理CAJ文件: {caj_path}")
                    try:
                        doc = fitz.open(caj_path)
                        text = ""
                        
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            page_text = page.get_text()
                            if page_text:
                                text += page_text + "\n\n"
                        
                        doc.close()
                        self.logger.info(f"成功从CAJ文件 '{caj_path}' 提取文本，共 {len(text)} 字符")
                        return text
                    except Exception as fitz_error:
                        # pymupdf可能不支持某些CAJ格式
                        self.logger.error(f"pymupdf无法处理CAJ文件: {str(fitz_error)}")
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
                self.logger.error(f"CAJ文件处理失败: {str(e)}")
                raise Exception(f"CAJ文件处理失败: {str(e)}")
        except Exception as e:
            self.logger.error(f"读取CAJ文件失败: {str(e)}")
            raise
    
    def read_document(self, file_path: str) -> str:
        """
        读取文档内容，支持多种格式
        
        Args:
            file_path: 文档文件路径
            
        Returns:
            str: 文档内容
            
        Raises:
            FileNotFoundError: 文件不存在
            ValueError: 不支持的文件格式
            Exception: 读取失败
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_extension == '.pdf':
                return self.read_pdf(file_path)
            elif file_extension == '.caj':
                return self.read_caj(file_path)
            elif file_extension in ['.txt', '.md']:
                # 文本文件直接读取
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                self.logger.info(f"成功读取文本文件 '{file_path}'，共 {len(content)} 字符")
                return content
            else:
                raise ValueError(f"不支持的文件格式: {file_extension}")
        except Exception as e:
            self.logger.error(f"读取文档 '{file_path}' 失败: {str(e)}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        提取文档元数据
        
        Args:
            file_path: 文档路径
            
        Returns:
            Dict[str, Any]: 包含元数据的字典
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 获取文件基本信息
        filename = os.path.basename(file_path)
        title = os.path.splitext(filename)[0]
        title = re.sub(r'[_-]+', ' ', title)  # 替换下划线和连字符为空格
        
        file_size = os.path.getsize(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        
        metadata = {
            'filename': filename,
            'title': title,
            'file_path': file_path,
            'file_size': file_size,
            'file_extension': file_extension,
        }
        
        # 如果是PDF文件，尝试提取更多元数据
        if file_extension == '.pdf':
            try:
                with open(file_path, 'rb') as file:
                    reader = PyPDF2.PdfReader(file)
                    metadata.update({
                        'num_pages': len(reader.pages),
                        'pdf_info': dict(reader.metadata) if reader.metadata else {}
                    })
                    
                    # 尝试从PDF元数据中提取标题
                    if reader.metadata and reader.metadata.get('/Title'):
                        metadata['title'] = reader.metadata['/Title']
                        
            except Exception as e:
                self.logger.warning(f"提取PDF元数据失败: {str(e)}")
        
        return metadata

# 创建全局实例
document_manager = DocumentManager()