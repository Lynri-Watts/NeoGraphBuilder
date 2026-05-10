#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ollama服务器连接诊断脚本
"""

import os
import sys
import requests
import json
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ollama_connection():
    """测试Ollama服务器连接"""
    
    # 读取配置文件
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    base_url = config.get('LLM_CONFIG', 'base_url')
    api_key = config.get('LLM_CONFIG', 'api_key')
    model = config.get('LLM_CONFIG', 'model_chat')
    
    logger.info(f"测试Ollama服务器连接")
    logger.info(f"服务器地址: {base_url}")
    logger.info(f"模型: {model}")
    
    # 测试基础连接
    try:
        # 移除/v1路径，测试基础API
        base_api_url = base_url.replace('/v1', '')
        
        # 测试Ollama原生API
        logger.info("测试Ollama原生API...")
        response = requests.get(f"{base_api_url}/api/tags", timeout=10)
        
        if response.status_code == 200:
            logger.info("✓ Ollama原生API连接成功")
            models = response.json().get('models', [])
            if models:
                logger.info(f"✓ 可用模型: {[model['name'] for model in models]}")
            else:
                logger.warning("⚠ 未找到可用模型")
        else:
            logger.error(f"✗ Ollama原生API连接失败: HTTP {response.status_code}")
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ 无法连接到Ollama服务器: {str(e)}")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"✗ 连接超时: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ 连接测试异常: {str(e)}")
        return False
    
    # 测试OpenAI兼容API
    try:
        logger.info("测试OpenAI兼容API...")
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 10
        }
        
        response = requests.post(f"{base_url}/chat/completions", 
                                headers=headers, 
                                json=payload, 
                                timeout=30)
        
        if response.status_code == 200:
            logger.info("✓ OpenAI兼容API连接成功")
            result = response.json()
            logger.info(f"✓ API响应正常")
            return True
        else:
            logger.error(f"✗ OpenAI兼容API连接失败: HTTP {response.status_code}")
            logger.error(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError as e:
        logger.error(f"✗ 无法连接到OpenAI兼容API: {str(e)}")
        return False
    except requests.exceptions.Timeout as e:
        logger.error(f"✗ OpenAI兼容API连接超时: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"✗ OpenAI兼容API测试异常: {str(e)}")
        return False

def test_network_connectivity():
    """测试网络连通性"""
    
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    base_url = config.get('LLM_CONFIG', 'base_url')
    
    # 提取主机和端口
    from urllib.parse import urlparse
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname
    port = parsed_url.port or 11434
    
    logger.info(f"测试网络连通性到 {host}:{port}")
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info("✓ 网络端口连通性正常")
            return True
        else:
            logger.error(f"✗ 网络端口无法连接: 错误代码 {result}")
            return False
            
    except Exception as e:
        logger.error(f"✗ 网络连通性测试异常: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始Ollama服务器连接诊断...")
    
    # 测试网络连通性
    network_ok = test_network_connectivity()
    
    # 测试API连接
    api_ok = test_ollama_connection()
    
    # 汇总结果
    logger.info("\n=== 诊断结果汇总 ===")
    
    if network_ok and api_ok:
        logger.info("✓ 所有测试通过 - Ollama服务器连接正常")
    else:
        logger.error("✗ 发现问题 - 请检查以下方面:")
        if not network_ok:
            logger.error("  - 网络连接问题")
        if not api_ok:
            logger.error("  - API连接问题")
        
        logger.info("\n建议检查:")
        logger.info("1. Ollama服务器是否正在运行")
        logger.info("2. 防火墙设置是否允许连接")
        logger.info("3. 服务器地址和端口是否正确")
        logger.info("4. 模型是否已正确加载")

if __name__ == "__main__":
    main()