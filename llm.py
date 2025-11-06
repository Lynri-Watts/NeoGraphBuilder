import json
import logging
import time
from typing import Dict, Any, Optional
from openai import OpenAI

# 配置日志
logger = logging.getLogger(__name__)

class LLMClient:
    """
    大语言模型客户端，提供单一开放接口处理大模型输入输出
    """
    
    def __init__(self, config: Dict[str, Any], extra_config: Optional[Dict[str, Any]] = None):
        """
        初始化LLM客户端
        
        Args:
            config: 大模型配置字典，包含api_key、base_url、model等
            extra_config: 额外配置参数
        """
        self.config = config
        self.extra_config = extra_config or {}
        
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url=config.get('base_url'),
            api_key=config.get('api_key')
        )
    
    def call_llm(self, prompt: str) -> str:
        """
        唯一的开放接口，调用大模型获取流式响应并解析
        
        Args:
            prompt: 完整的提示词内容
            
        Returns:
            大模型回答的内容字符串
        """
        try:
            logger.info("开始调用大模型...")
            start_time = time.time()
            
            # 调用大模型获取流式响应
            response = self.client.chat.completions.create(
                model=self.config["model"],
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