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
        只负责解析第一层JSON，获取大模型的回答内容
        
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
            
            # 处理流式响应 - 正确解析每个chunk的JSON
            think_str = ""
            answer_str = ""
            last_progress_length = 0  # 记录上一次显示进度时的长度
            progress_interval = 500  # 进度显示间隔（字符数）
            
            for chunk in response:
                content = chunk.choices[0].delta.content or ""
                if content:
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

            
            logger.info(f"最终回答内容前200字符: {answer_str[:200]}...")
            
            return answer_str
            
        except Exception as e:
            logger.error(f"大模型调用失败: {str(e)}")
            raise    