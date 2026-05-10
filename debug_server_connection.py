#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
服务器连接详细诊断脚本
"""

import os
import sys
import requests
import socket
import json
import logging
from urllib.parse import urlparse

# 配置详细日志
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_server_details():
    """检查服务器详细信息"""
    
    import configparser
    config = configparser.ConfigParser()
    config.read('config.ini')
    
    base_url = config.get('LLM_CONFIG', 'base_url')
    api_key = config.get('LLM_CONFIG', 'api_key')
    model = config.get('LLM_CONFIG', 'model_chat')
    
    logger.info("=== 服务器配置信息 ===")
    logger.info(f"服务器地址: {base_url}")
    logger.info(f"API密钥: {api_key[:10]}...{api_key[-10:] if len(api_key) > 20 else '***'}")
    logger.info(f"模型名称: {model}")
    
    # 解析URL
    parsed = urlparse(base_url)
    logger.info(f"协议: {parsed.scheme}")
    logger.info(f"主机: {parsed.hostname}")
    logger.info(f"端口: {parsed.port}")
    logger.info(f"路径: {parsed.path}")
    
    return parsed.hostname, parsed.port or 11434

def test_network_connectivity(host, port):
    """测试网络连通性"""
    
    logger.info("\n=== 网络连通性测试 ===")
    
    # 测试DNS解析
    try:
        ip_address = socket.gethostbyname(host)
        logger.info(f"✓ DNS解析成功: {host} -> {ip_address}")
    except socket.gaierror as e:
        logger.error(f"✗ DNS解析失败: {str(e)}")
        return False
    
    # 测试端口连通性
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(10)
        result = sock.connect_ex((host, port))
        sock.close()
        
        if result == 0:
            logger.info(f"✓ 端口 {port} 连通性正常")
            return True
        else:
            logger.error(f"✗ 端口 {port} 无法连接: 错误代码 {result}")
            
            # 常见错误代码解释
            error_codes = {
                10060: "连接超时",
                10061: "连接被拒绝（服务器未启动或防火墙阻止）",
                10054: "连接被重置",
                10065: "目标主机不可达"
            }
            
            if result in error_codes:
                logger.error(f"  可能原因: {error_codes[result]}")
            
            return False
            
    except Exception as e:
        logger.error(f"✗ 网络连通性测试异常: {str(e)}")
        return False

def test_ollama_api(host, port, base_url, api_key, model):
    """测试Ollama API"""
    
    logger.info("\n=== Ollama API测试 ===")
    
    # 测试基础API端点
    endpoints = [
        (f"http://{host}:{port}/api/tags", "模型列表"),
        (f"http://{host}:{port}/api/version", "版本信息"),
        (base_url + "/models", "OpenAI兼容模型列表"),
        (base_url + "/chat/completions", "OpenAI兼容聊天接口")
    ]
    
    for endpoint, description in endpoints:
        try:
            logger.info(f"测试 {description}: {endpoint}")
            
            if "chat/completions" in endpoint:
                # 测试OpenAI兼容API
                headers = {'Content-Type': 'application/json'}
                if api_key:
                    headers['Authorization'] = f'Bearer {api_key}'
                
                payload = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
                
                response = requests.post(endpoint, headers=headers, json=payload, timeout=15)
            else:
                # 测试GET端点
                response = requests.get(endpoint, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✓ {description} 连接成功")
                
                if "api/tags" in endpoint:
                    models = response.json().get('models', [])
                    if models:
                        model_names = [m['name'] for m in models]
                        logger.info(f"  可用模型: {model_names}")
                        if model not in model_names:
                            logger.warning(f"⚠ 配置的模型 '{model}' 不在可用模型列表中")
                    else:
                        logger.warning("⚠ 未找到可用模型")
                
                if "api/version" in endpoint:
                    version_info = response.json()
                    logger.info(f"  Ollama版本: {version_info}")
                    
            else:
                logger.error(f"✗ {description} 连接失败: HTTP {response.status_code}")
                logger.error(f"  响应内容: {response.text[:200]}")
                
        except requests.exceptions.ConnectionError as e:
            logger.error(f"✗ {description} 连接错误: {str(e)}")
        except requests.exceptions.Timeout as e:
            logger.error(f"✗ {description} 连接超时: {str(e)}")
        except Exception as e:
            logger.error(f"✗ {description} 测试异常: {str(e)}")

def check_firewall_and_proxy():
    """检查防火墙和代理设置"""
    
    logger.info("\n=== 防火墙和代理检查 ===")
    
    # 检查系统代理
    try:
        proxy_env_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        proxies_set = False
        
        for var in proxy_env_vars:
            if os.environ.get(var):
                logger.info(f"检测到代理设置: {var}={os.environ.get(var)[:50]}...")
                proxies_set = True
        
        if not proxies_set:
            logger.info("✓ 未检测到代理设置")
        
    except Exception as e:
        logger.error(f"代理检查异常: {str(e)}")

def test_alternative_ports(host):
    """测试备用端口"""
    
    logger.info("\n=== 备用端口测试 ===")
    
    common_ports = [11434, 8080, 8000, 3000, 5000, 7860]
    
    for port in common_ports:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            
            if result == 0:
                logger.info(f"✓ 端口 {port} 开放")
            else:
                logger.debug(f"端口 {port} 关闭")
                
        except Exception as e:
            logger.debug(f"端口 {port} 测试异常: {str(e)}")

def main():
    """主函数"""
    
    logger.info("开始服务器连接详细诊断...")
    
    try:
        # 获取服务器信息
        host, port = check_server_details()
        
        # 读取配置
        import configparser
        config = configparser.ConfigParser()
        config.read('config.ini')
        base_url = config.get('LLM_CONFIG', 'base_url')
        api_key = config.get('LLM_CONFIG', 'api_key')
        model = config.get('LLM_CONFIG', 'model_chat')
        
        # 执行各项测试
        check_firewall_and_proxy()
        network_ok = test_network_connectivity(host, port)
        
        if network_ok:
            test_ollama_api(host, port, base_url, api_key, model)
        else:
            test_alternative_ports(host)
        
        # 汇总结果
        logger.info("\n=== 诊断结果汇总 ===")
        
        if not network_ok:
            logger.error("✗ 主要问题: 网络连接失败")
            logger.info("\n建议解决方案:")
            logger.info("1. 检查Ollama服务器是否正在运行")
            logger.info("2. 确认服务器地址和端口是否正确")
            logger.info("3. 检查防火墙设置")
            logger.info("4. 联系服务器管理员确认服务状态")
            logger.info("5. 考虑使用本地Ollama安装作为备选方案")
        else:
            logger.info("✓ 网络连接正常，请检查API接口")
            
    except Exception as e:
        logger.error(f"诊断过程异常: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()