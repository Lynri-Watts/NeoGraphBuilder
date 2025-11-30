import json
import logging
import time
import configparser
import os
from typing import Dict, Any, Optional
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)

def load_llm_config(config_file: str = "config.ini") -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    从配置文件加载LLM配置
    
    Args:
        config_file: 配置文件路径
        
    Returns:
        tuple: (LLM_CONFIG, LLM_EXTRA_CONFIG)
    """
    config = configparser.ConfigParser()
    
    # 检查配置文件是否存在
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"配置文件 '{config_file}' 不存在")
    
    config.read(config_file)
    
    # 读取LLM基础配置
    llm_config = {
        "base_url": config.get('LLM_CONFIG', 'base_url'),
        "api_key": config.get('LLM_CONFIG', 'api_key'),
        "model_reasoner": config.get('LLM_CONFIG', 'model_reasoner'),
        "model_chat": config.get('LLM_CONFIG', 'model_chat')
    }
    
    # 读取LLM额外配置
    llm_extra_config = {}
    if 'LLM_EXTRA_CONFIG' in config:
        for key, value in config.items('LLM_EXTRA_CONFIG'):
            try:
                # 尝试解析为布尔值
                llm_extra_config[key] = config.getboolean('LLM_EXTRA_CONFIG', key)
            except ValueError:
                # 如果不是布尔值，直接使用字符串值
                llm_extra_config[key] = value
        logger.debug(f"加载的LLM_EXTRA_CONFIG: {llm_extra_config}")
    else:
        logger.warning("配置文件中未找到LLM_EXTRA_CONFIG部分")
    
    logger.info("LLM配置加载完成")
    return llm_config, llm_extra_config

class LLMClient:
    """
    大语言模型客户端，提供单一开放接口处理大模型输入输出
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, extra_config: Optional[Dict[str, Any]] = None, config_file: str = "config.ini"):
        """
        初始化LLM客户端
        
        Args:
            config: 大模型配置字典，包含api_key、base_url、model等（可选）
            extra_config: 额外配置参数（可选）
            config_file: 配置文件路径（当config为None时使用）
        """
        # 如果没有提供配置，则从配置文件读取
        if config is None:
            self.config, self.extra_config = load_llm_config(config_file)
        else:
            self.config = config
            self.extra_config = extra_config or {}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=self.config.get('base_url'),
            api_key=self.config.get('api_key')
        )
    
    def call_llm(self, prompt: str, model_type: str = "reasoner") -> str:
        """
        唯一的开放接口，调用大模型获取流式响应并解析
        
        Args:
            prompt: 完整的提示词内容
            model_type: 模型类型，可选值为 "reasoner" 或 "chat"
            
        Returns:
            大模型回答的内容字符串
        """
        try:
            logger.info(f"开始调用大模型（模型类型: {model_type}）...")
            start_time = time.time()
            
            # 根据模型类型选择模型
            if model_type == "chat":
                model = self.config.get("model_chat", self.config.get("model_reasoner", self.config.get("model")))
            else:  # 默认使用reasoner
                model = self.config.get("model_reasoner", self.config.get("model"))
            
            # 调用大模型获取流式响应
            response = self.client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                extra_body=self.extra_config
            )
            
            # 累积所有响应内容
            full_response = ""
            last_progress_length = 0  # 记录上一次显示进度时的长度
            progress_interval = 500  # 进度显示间隔（字符数）
            
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
                    full_response += content
                    
                    # 实时打印处理进度
                    if len(full_response) > last_progress_length + progress_interval:
                        logger.info(f"大模型响应处理中...已累积{len(full_response)}字符")
                        last_progress_length = len(full_response)
            
            logger.info(f"大模型调用完成，总响应长度: {len(full_response)}字符")
            
            # 尝试解析完整的响应内容
            try:
                # 检查是否需要特殊处理（如果是知识图谱格式的响应）
                if full_response.strip().startswith('{') and '"nodes"' in full_response and '"relations"' in full_response:
                    # 这已经是知识图谱的JSON格式，直接返回
                    return full_response
                else:
                    # 尝试解析为JSON看是否有msg字段
                    try:
                        response_data = json.loads(full_response)
                        if isinstance(response_data, dict) and 'msg' in response_data:
                            return response_data['msg']
                    except:
                        # 如果不是JSON或没有msg字段，直接返回原始响应
                        pass
                    
                    # 最终返回原始响应内容
                    return full_response
            except json.JSONDecodeError as e:
                logger.error(f"JSON解析失败: {str(e)}，将返回原始响应内容")
                logger.debug(f"原始响应内容: {full_response}")
                # 返回原始内容，让上层处理
                return full_response
            
        except Exception as e:
            logger.error(f"大模型调用失败: {str(e)}")
            # 记录更详细的错误信息
            logger.debug(f"错误类型: {type(e).__name__}")
            raise