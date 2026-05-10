#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基于知识图谱的RAG问答系统 - Gradio界面

该脚本为query.py中的知识图谱问答系统提供一个Web界面，使用户可以通过浏览器与系统交互。
"""

import os
import logging
import gradio as gr
from query import KnowledgeGraphQA, logger

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量，用于存储问答系统实例
qa_system = None


def initialize_qa_system():
    """初始化知识图谱问答系统"""
    global qa_system
    if qa_system is None:
        try:
            logger.info("正在初始化知识图谱问答系统...")
            qa_system = KnowledgeGraphQA()
            logger.info("知识图谱问答系统初始化完成")
            return "✅ 系统初始化成功"
        except Exception as e:
            logger.error(f"系统初始化失败: {str(e)}")
            return f"❌ 系统初始化失败: {str(e)}"
    return "✅ 系统已经初始化"


def answer_with_system(query, top_k=5):
    """
    使用问答系统回答用户问题
    
    Args:
        query: 用户查询文本
        top_k: 返回的最大相关项数
        
    Returns:
        生成的回答文本
    """
    # 检查系统是否已初始化
    global qa_system
    if qa_system is None:
        init_status = initialize_qa_system()
        if "失败" in init_status:
            return f"系统未初始化: {init_status}"
    
    try:
        if not query or query.strip() == "":
            return "请输入有效的问题"
        
        # 调用问答系统回答问题
        answer = qa_system.answer_query(query)
        return answer
    except Exception as e:
        logger.error(f"回答问题时出错: {str(e)}")
        return f"处理问题时发生错误: {str(e)}"


def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="知识图谱智能问答系统") as demo:
        # 页面标题
        gr.Markdown("""
        # 知识图谱智能问答系统
        
        基于Neo4j知识图谱的智能问答助手，能够从构建的知识图谱中检索相关信息并生成准确的回答。
        
        **使用说明：**
        1. 确保系统已初始化
        2. 在下方输入框中输入您的问题
        3. 点击"生成回答"按钮获取答案
        """)
        
        # 系统状态和初始化按钮
        status_text = gr.Textbox(label="系统状态", value="等待初始化...", interactive=False)
        init_btn = gr.Button("初始化系统", variant="primary")
        init_btn.click(initialize_qa_system, outputs=status_text)
        
        # 查询输入和参数设置
        with gr.Row():
            with gr.Column(scale=1):
                top_k_slider = gr.Slider(
                    minimum=1, maximum=15, value=5, step=1,
                    label="检索相关信息数量",
                    info="设置从知识图谱中检索的相关信息数量"
                )
            
            with gr.Column(scale=3):
                query_input = gr.Textbox(
                    placeholder="请输入您的问题...",
                    label="问题",
                    lines=3,
                    show_label=True
                )
        
        # 生成回答按钮
        generate_btn = gr.Button("生成回答", variant="secondary", size="lg")
        
        # 进度条组件
        progress_bar = gr.Slider(
            minimum=0, maximum=100, value=0, step=1, visible=False, 
            label="生成进度", interactive=False
        )
        
        # 回答输出
        answer_output = gr.Markdown(
            label="回答",
            value="<div style='color:gray; font-style:italic;'>回答将显示在这里...</div>"
        )
        
        # 设置点击事件
        def on_generate_click(query, top_k):
            # 检查输入
            if not query or query.strip() == "":
                yield {answer_output: "请输入有效的问题", progress_bar: gr.update(value=0, visible=False)}
                return
            
            # 显示进度条并设置初始进度
            yield {answer_output: "<div style='color:gray; font-style:italic;'>正在处理您的问题...</div>", progress_bar: gr.update(value=0, visible=True)}
            
            # 模拟进度更新 - 初始化阶段
            yield {answer_output: "<div style='color:gray; font-style:italic;'>正在初始化问答系统...</div>", progress_bar: gr.update(value=20, visible=True)}
            
            # 检查系统是否已初始化
            global qa_system
            if qa_system is None:
                init_status = initialize_qa_system()
                if "失败" in init_status:
                    yield {answer_output: f"系统未初始化: {init_status}", progress_bar: gr.update(value=0, visible=False)}
                    return
            
            # 模拟进度更新 - 检索阶段
            yield {answer_output: "<div style='color:gray; font-style:italic;'>正在从知识图谱检索相关信息...</div>", progress_bar: gr.update(value=50, visible=True)}
            
            try:
                # 模拟进度更新 - 生成回答阶段
                yield {answer_output: "<div style='color:gray; font-style:italic;'>正在生成回答...</div>", progress_bar: gr.update(value=75, visible=True)}
                
                # 调用问答系统回答问题
                answer = qa_system.answer_query(query)
                
                # 完成进度
                yield {answer_output: answer, progress_bar: gr.update(value=100, visible=True)}
                
                # 短暂显示100%进度后隐藏
                import time
                time.sleep(0.5)
                yield {answer_output: answer, progress_bar: gr.update(value=0, visible=False)}
                
            except Exception as e:
                logger.error(f"回答问题时出错: {str(e)}")
                error_msg = f"处理问题时发生错误: {str(e)}"
                yield {answer_output: error_msg, progress_bar: gr.update(value=0, visible=False)}
        
        # 连接按钮和函数 - 支持多输出
        generate_btn.click(
            fn=on_generate_click,
            inputs=[query_input, top_k_slider],
            outputs=[answer_output, progress_bar]
        )
        
        # 快捷键支持
        query_input.submit(
            fn=on_generate_click,
            inputs=[query_input, top_k_slider],
            outputs=[answer_output, progress_bar]
        )
        
        # 底部信息
        gr.Markdown("""
        --- 
        
        *系统提示：*
        - 请确保Neo4j数据库服务正在运行
        - 提问时尽量使用具体、明确的关键词
        - 对于复杂问题，可能需要多次调整提问方式
        """)
    
    return demo


def main():
    """主函数，启动Gradio界面"""
    print("=== 知识图谱RAG问答系统 - Web界面 ===")
    print("正在启动Gradio界面...")
    
    # 创建并启动界面
    demo = create_gradio_interface()
    
    # 启动界面
    try:
        demo.launch(
            server_name="0.0.0.0",  # 允许从任何IP访问
            server_port=7861,       # 修改为端口7861避免冲突
            share=False,            # 不生成公开链接
            debug=False
        )
    except KeyboardInterrupt:
        print("\n用户中断，正在关闭...")
    except Exception as e:
        print(f"启动界面时出错: {str(e)}")
    finally:
        # 确保关闭数据库连接
        global qa_system
        if qa_system is not None:
            qa_system.close()
            print("已关闭知识图谱连接")


if __name__ == "__main__":
    main()