#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器状态检查脚本
"""

import socket
import requests
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_server_status():
    """检查服务器状态"""
    
    server_host = "202.205.127.227"
    server_port = 11434
    
    logger.info(f"检查服务器状态: {server_host}:{server_port}")
    
    # 1. 测试网络连通性
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((server_host, server_port))
        sock.close()
        
        if result == 0:
            logger.info("✓ 网络连通性正常")
        else:
            logger.error(f"✗ 网络连接失败，错误代码: {result}")
            logger.info("建议: 检查服务器是否启动，防火墙设置")
            return False
    except Exception as e:
        logger.error(f"✗ 网络测试异常: {str(e)}")
        return False
    
    # 2. 测试Ollama API
    try:
        response = requests.get(f"http://{server_host}:{server_port}/api/tags", timeout=10)
        if response.status_code == 200:
            logger.info("✓ Ollama API响应正常")
            models = response.json().get('models', [])
            if models:
                logger.info(f"✓ 可用模型: {[m['name'] for m in models]}")
            return True
        else:
            logger.error(f"✗ API响应异常: HTTP {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        logger.error("✗ 无法连接到Ollama API")
        return False
    except requests.exceptions.Timeout:
        logger.error("✗ API请求超时")
        return False
    except Exception as e:
        logger.error(f"✗ API测试异常: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始服务器状态检查...")
    
    if check_server_status():
        logger.info("\n=== 服务器状态正常 ===")
        logger.info("建议: 检查config.ini配置文件中的API密钥和模型名称")
    else:
        logger.error("\n=== 服务器连接失败 ===")
        logger.info("请按照以下步骤排查:")
        logger.info("1. 联系服务器管理员确认服务状态")
        logger.info("2. 检查服务器防火墙设置")
        logger.info("3. 确认Ollama服务已启动并监听正确端口")
        logger.info("4. 考虑使用本地Ollama安装作为备选方案")

if __name__ == "__main__":
    main()