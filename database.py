from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import os
import time
import logging
import traceback
import json

# 使用已配置的logger
logger = logging.getLogger(__name__)

class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        # 配置模型缓存路径
        model_cache_dir = os.path.join(os.path.dirname(__file__), 'models')
        os.makedirs(model_cache_dir, exist_ok=True)
        
        # 使用本地缓存加载多语言模型，更好地支持人名、缩写和不常见概念
        model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
        
        # 先检查本地是否存在模型
        try:
            # 尝试只从本地加载模型
            self.model = SentenceTransformer(model_name, 
                                          cache_folder=model_cache_dir,
                                          local_files_only=True)
            logger.info(f"成功从本地加载模型: {model_name}")
        except (OSError, FileNotFoundError):
            # 如果本地不存在，则从网络下载
            logger.info(f"本地模型不存在，开始从网络下载: {model_name}")
            self.model = SentenceTransformer(model_name, 
                                          cache_folder=model_cache_dir,
                                          local_files_only=False)
            logger.info(f"模型下载完成: {model_name}")
        
        # 检查并加载Spacy模型
        spacy_model = "en_core_web_sm"
        try:
            # 尝试加载Spacy模型
            self.nlp = spacy.load(spacy_model)
            logger.info(f"成功加载Spacy模型: {spacy_model}")
        except OSError:
            # 如果模型不存在，尝试下载
            logger.info(f"Spacy模型不存在，开始下载: {spacy_model}")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", spacy_model])
            self.nlp = spacy.load(spacy_model)
            logger.info(f"Spacy模型下载完成: {spacy_model}")
        
        # 初始化数据库索引
        self.__initialize_database_indexes()
        
        # 初始化统计变量
        self.stats = {
            'similarity_matches': {
                'exact_match': 0,  # 精确匹配
                'alias_match': 0,  # 别名匹配
                'fuzzy_match': 0,  # 模糊匹配
                'vector_match': 0,  # 向量相似度匹配
                'context_match': 0  # 上下文匹配
            },
            'nodes': {
                'new_concepts': 0,
                'merged_concepts': 0,
                'new_entities': 0,
                'merged_entities': 0
            },
            'relations': {
                'new_relations': 0,
                'merged_relations': 0
            }
        }
    
    def __initialize_database_indexes(self):
        """创建必要的数据库索引"""
        with self.driver.session() as session:
            # 创建Concept类型的索引
            # 文本索引 - 分别为每个属性创建索引
            session.run("""
            CREATE TEXT INDEX concept_name_index IF NOT EXISTS 
            FOR (c:Concept) ON (c.name)
            """)
            
            # session.run("""
            # CREATE TEXT INDEX concept_canonical_name_index IF NOT EXISTS 
            # FOR (c:Concept) ON (c.canonicalName)
            # """)
            
            session.run("""
            CREATE TEXT INDEX concept_aliases_index IF NOT EXISTS 
            FOR (c:Concept) ON (c.aliases)
            """)
            
            # 检查并创建Concept向量索引（如果不存在）
            check_query = """
            SHOW INDEXES 
            WHERE name = 'concept_vector_index' 
            AND type = 'VECTOR' 
            AND entityType = 'NODE' 
            AND labelsOrTypes = ['Concept'] 
            AND properties = ['embedding']
            """
            result = session.run(check_query)
            if not list(result):
                session.run("""
                CALL db.index.vector.createNodeIndex(
                    'concept_vector_index',
                    'Concept',
                    'embedding',
                    384,
                    'cosine'
                )
                """)
            
            # 创建Entity类型的索引
            # 文本索引
            session.run("""
            CREATE TEXT INDEX entity_name_index IF NOT EXISTS 
            FOR (e:Entity) ON (e.name)
            """)
            
            # session.run("""
            # CREATE TEXT INDEX entity_canonical_name_index IF NOT EXISTS 
            # FOR (e:Entity) ON (e.canonicalName)
            # """)
            
            session.run("""
            CREATE TEXT INDEX entity_aliases_index IF NOT EXISTS 
            FOR (e:Entity) ON (e.aliases)
            """)
            
            # 检查并创建Entity向量索引（如果不存在）
            check_query = """
            SHOW INDEXES 
            WHERE name = 'entity_vector_index' 
            AND type = 'VECTOR' 
            AND entityType = 'NODE' 
            AND labelsOrTypes = ['Entity'] 
            AND properties = ['embedding']
            """
            result = session.run(check_query)
            if not list(result):
                session.run("""
                CALL db.index.vector.createNodeIndex(
                    'entity_vector_index',
                    'Entity',
                    'embedding',
                    384,
                    'cosine'
                )
                """)
            
            # 创建关系索引
            session.run("""
            CREATE INDEX rel_index IF NOT EXISTS 
            FOR ()-[r:RELATED_TO]-() ON (r.type)
            """)
            
            logger.info("Database indexes initialized for Concept and Entity types")
    
    def __generate_vector(self, text):
        """为文本生成嵌入向量，采用逐个编码方式以获得更准确的向量表示"""
        import time
        start_time = time.time()
        
        # 确保文本是字符串类型
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"无效文本输入: {text}")
            # 返回一个零向量作为默认值
            return [0.0] * self.model.get_sentence_embedding_dimension()
            
        try:
            # 记录编码前的详细信息
            text_length = len(text)
            text_preview = text[:100] + "..." if text_length > 100 else text
            logger.info(f"[SENTENCE_TRANSFORMER] 开始编码文本 - 长度: {text_length}, 预览: '{text_preview}'")
            
            # 逐一生成向量，确保更准确的编码
            encode_start = time.time()
            vector = self.model.encode(text, convert_to_numpy=True)
            encode_time = time.time() - encode_start
            
            # 记录编码后的信息
            logger.info(f"[SENTENCE_TRANSFORMER] 编码完成 - 耗时: {encode_time:.3f}s, 向量维度: {vector.shape if hasattr(vector, 'shape') else len(vector)}")
            
            # 验证向量是否有效
            if vector is None or len(vector) == 0:
                logger.warning(f"生成的向量为空: {text}")
                return [0.0] * self.model.get_sentence_embedding_dimension()
                
            # 检查向量中是否有NaN值
            if np.isnan(vector).any():
                logger.warning(f"生成的向量包含NaN值: {text}")
                return [0.0] * self.model.get_sentence_embedding_dimension()
            
            total_time = time.time() - start_time
            logger.info(f"[SENTENCE_TRANSFORMER] __generate_vector完成 - 总耗时: {total_time:.3f}s")
                
            return vector.tolist()
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"[SENTENCE_TRANSFORMER] 生成向量时出错 - 耗时: {error_time:.3f}s, 文本: '{text[:50]}...', 错误: {str(e)}")
            return [0.0] * self.model.get_sentence_embedding_dimension()
    
    def __find_similar_nodes(self, node):
        """在Neo4j中只查找名称完全相同的节点"""
        node_type = node.get("type", "Concept")
        label = "Concept" if node_type == "Concept" else "Entity"
        
        with self.driver.session() as session:
            # 只进行精确名称匹配
            exact_match_query = f"""
            MATCH (n:{label})
            WHERE n.name = $name 
            RETURN n.id AS node_id, n.name AS name
            """
            exact_results = list(session.run(exact_match_query, 
                                           name=node["name"]))
            
            if exact_results:
                return exact_results
            
            # 没有找到名称完全相同的节点
            return []
    
    def __merge_or_create_node(self, node, candidates):
        """只在名称完全相同时合并节点，否则创建新节点"""
        node_type = node.get("type", "Concept")
        
        # 如果没有候选，直接创建新节点
        if not candidates:
            node_id = self.__create_new_node(node)
            return {"node_id": node_id, "is_new": True}
        
        # 只有精确匹配（相似度为1.0）才合并
        best_candidate = candidates[0]
        # if best_candidate["similarity"] == 1.0:
        if node_type == "Concept":
            self.stats['nodes']['merged_concepts'] += 1
        else:
            self.stats['nodes']['merged_entities'] += 1
        node_id = self.__merge_node(best_candidate["node_id"], node)
        return {"node_id": node_id, "is_new": False}
        
        # # 不符合合并条件，创建新节点
        # if node_type == "Concept":
        #     self.stats['nodes']['new_concepts'] += 1
        # else:
        #     self.stats['nodes']['new_entities'] += 1
        # node_id = self.__create_new_node(node)
        # return {"node_id": node_id, "is_new": True}
    
    # __check_context_similarity方法已移除，不再需要检测上下文相似度
    
    def __create_new_node(self, node):
        """创建新节点，根据类型创建Concept或Entity，不再存储别名列表"""
        node_type = node.get("type", "Concept")
        
        # 获取description，默认为空字符串
        description = node.get("description", "")
            
        query = f"""
        CREATE (n:{node_type} {{
            id: apoc.create.uuid(),
            name: $name,
            description: $description,
            embedding: $vector,
            sourceDocuments: $source_documents,
            type: $node_type
        }})
        RETURN n.id AS node_id
        """
        
        # 使用私有方法处理source_document字段
        source_documents = self.__prepare_source_documents(node)
        
        # 将字典列表转换为JSON字符串列表，符合Neo4j属性类型要求
        source_documents_json = [json.dumps(doc) for doc in source_documents]
        
        with self.driver.session() as session:
            result = session.run(query, 
                name=node["name"],
                canonical_name=node["canonical_name"],
                description=description,
                vector=node["vector"],
                source_documents=source_documents_json,
                node_type=node_type
            )
            record = result.single()
            return record["node_id"]
    
    def __merge_node(self, existing_id, new_node):
        """合并到现有节点，不再合并别名列表，按文件名合并source_documents"""
        # 获取新节点的description，默认为空字符串
        new_description = new_node.get("description", "")
        
        # 使用私有方法处理source_document字段
        new_source_documents = self.__prepare_source_documents(new_node)
        
        # 获取现有节点的sourceDocuments并合并去重
        with self.driver.session() as session:
            # 先获取现有的sourceDocuments
            get_existing_query = """
            MATCH (n) 
            WHERE n.id = $existing_id AND (n:Concept OR n:Entity)
            RETURN n.sourceDocuments AS existing_docs
            """
            result = session.run(get_existing_query, existing_id=existing_id)
            record = result.single()
            existing_docs_json = record["existing_docs"] if record and record["existing_docs"] else []
            
            # 将现有的JSON字符串转换为字典列表
            existing_docs = []
            for doc_json in existing_docs_json:
                try:
                    doc = json.loads(doc_json)
                    if isinstance(doc, dict):
                        existing_docs.append(doc)
                except (json.JSONDecodeError, TypeError):
                    continue
            
            # 合并新文档，按文件名、标题、段落编号组合去重
            merged_docs = existing_docs.copy()
            for new_doc in new_source_documents:
                # 检查是否已存在相同的文档
                is_duplicate = False
                for existing_doc in existing_docs:
                    if (existing_doc.get('filename') == new_doc.get('filename') and
                        existing_doc.get('title') == new_doc.get('title') and
                        existing_doc.get('paragraph_number') == new_doc.get('paragraph_number')):
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    merged_docs.append(new_doc)
            
            # 将合并后的文档列表转换为JSON字符串列表
            merged_docs_json = [json.dumps(doc) for doc in merged_docs]
            
            # 更新节点
            update_query = """
            MATCH (n) 
            WHERE n.id = $existing_id AND (n:Concept OR n:Entity)
            SET n.description = CASE WHEN $new_description <> '' 
                                    THEN CASE WHEN n.description IS NOT NULL AND n.description <> '' 
                                              THEN n.description + '\n' + $new_description
                                              ELSE $new_description
                                         END
                                    ELSE n.description END,
                n.sourceDocuments = $merged_docs_json
            RETURN n.id AS node_id
            """
            
            result = session.run(update_query, 
                existing_id=existing_id,
                new_description=new_description,
                merged_docs_json=merged_docs_json
            )
            record = result.single()
            return record["node_id"]
    
    def insert_nodes(self, nodes):
        """插入节点到知识图谱
        
        Args:
            nodes (list): 节点列表，每个节点为字典，包含name和type
                           可选字段：description（描述）、context（上下文列表）和source_document（来源文档）
            
        Returns:
            int: 成功处理的节点数量
        """
        processed_count = 0
        new_concepts_count = 0  # 新增Concept计数
        merged_concepts_count = 0  # 合并Concept计数
        new_entities_count = 0  # 新增Entity计数
        merged_entities_count = 0  # 合并Entity计数
        
        # 处理每个节点
        for node in nodes:
            try:
                # 验证必要字段
                if 'name' not in node or 'type' not in node:
                    logger.warning(f"节点缺少必要字段: {node}")
                    continue
                
                # 验证type字段的有效性
                if node['type'] not in ['Concept', 'Entity']:
                    logger.warning(f"节点类型无效: {node['type']}，默认为Concept")
                    node['type'] = 'Concept'
                
                # 1. 计算词向量
                node_vector = self.__generate_vector(node['name'])
                
                # 准备完整的节点数据，包含context、source_document和aliases
                full_node = {
                    'name': node['name'],
                    'canonical_name': self.nlp(node['name'])[0].lemma_.lower() if self.nlp(node['name']) else node['name'].lower(),
                    'description': node.get('description', ''),  # description现在是可选的，默认为空字符串
                    'type': node['type'],
                    'vector': node_vector,
                    'context': node.get('context', []),  # 使用节点中提供的上下文，默认为空列表
                    'source_document': node.get('source_document', 'manual_insert'),  # 使用节点中提供的来源文档，默认为manual_insert
                    'aliases': node.get('aliases', [])  # 使用节点中提供的别名列表，默认为空列表
                }
                
                logger.info(f"处理节点: '{node['name']}' (类型: {node['type']}) 及其别名: {full_node['aliases']}")
                
                # 2. 查找相似节点
                similar_nodes = self.__find_similar_nodes(full_node)
                
                # 3. 合并或创建新节点
                primary_node_id = self.__merge_or_create_node(full_node, similar_nodes)
                
                # 统计新增和合并的节点，按类型分类
                if node['type'] == 'Concept':
                    if similar_nodes:  # 有相似节点说明找到了精确匹配，需要合并
                        merged_concepts_count += 1
                        logger.info(f"已合并概念 '{node['name']}' -> ID: {primary_node_id}")
                    else:
                        new_concepts_count += 1
                        logger.info(f"已新增概念 '{node['name']}' -> ID: {primary_node_id}")
                else:  # Entity
                    if similar_nodes:  # 有相似节点说明找到了精确匹配，需要合并
                        merged_entities_count += 1
                        logger.info(f"已合并实体 '{node['name']}' -> ID: {primary_node_id}")
                    else:
                        new_entities_count += 1
                        logger.info(f"已新增实体 '{node['name']}' -> ID: {primary_node_id}")
                
                # 处理别名列表，为每个别名创建新节点并建立IS_ALIAS关系
                aliases = node.get('aliases', [])
                source_document = node.get('source_document', 'manual_insert')
                
                for alias in aliases:
                    # 跳过与主名称相同的别名
                    if alias == node['name']:
                        continue
                    
                    # 创建别名节点
                    alias_node = {
                        'name': alias,
                        'type': node['type'],
                        'description': f"别名: {node['name']}",
                        'source_document': source_document,
                        'canonical_name': self.nlp(alias)[0].lemma_.lower() if self.nlp(alias) else alias.lower(),
                        'vector': self.__generate_vector(alias),
                        'context': node.get('context', [])
                    }
                    
                    # 查找别名节点是否已存在（精确匹配）
                    alias_similar_nodes = self.__find_similar_nodes(alias_node)
                    alias_node_id = self.__merge_or_create_node(alias_node, alias_similar_nodes)
                    
                    # 更新统计信息
                    if node['type'] == 'Concept':
                        if alias_similar_nodes:  # 有相似节点说明找到了精确匹配，需要合并
                            merged_concepts_count += 1
                            logger.info(f"已合并别名概念 '{alias}' -> ID: {alias_node_id}")
                        else:
                            new_concepts_count += 1
                            logger.info(f"已新增别名概念 '{alias}' -> ID: {alias_node_id}")
                    else:  # Entity
                        if alias_similar_nodes:  # 有相似节点说明找到了精确匹配，需要合并
                            merged_entities_count += 1
                            logger.info(f"已合并别名实体 '{alias}' -> ID: {alias_node_id}")
                        else:
                            new_entities_count += 1
                            logger.info(f"已新增别名实体 '{alias}' -> ID: {alias_node_id}")
                    
                    # 创建双向IS_ALIAS关系
                    # 提取实际的节点ID字符串
                    primary_id = primary_node_id['node_id'] if isinstance(primary_node_id, dict) else primary_node_id
                    alias_id = alias_node_id['node_id'] if isinstance(alias_node_id, dict) else alias_node_id
                    self.__create_alias_relations(primary_id, alias_id, source_document)
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理节点 '{node.get('name', 'Unknown')}' 时出错: {str(e)}")
                logger.error(f"完整错误堆栈:\n{traceback.format_exc()}")
                continue
        
        # 输出最终统计信息
        logger.info(f"节点处理完成: 总共处理 {processed_count} 个节点")
        logger.info(f"概念统计: 新增 {new_concepts_count} 个，合并 {merged_concepts_count} 个")
        logger.info(f"实体统计: 新增 {new_entities_count} 个，合并 {merged_entities_count} 个")
        
        # 更新全局统计变量（累加而不是覆盖）
        self.stats['nodes']['new_concepts'] += new_concepts_count
        self.stats['nodes']['merged_concepts'] += merged_concepts_count
        self.stats['nodes']['new_entities'] += new_entities_count
        self.stats['nodes']['merged_entities'] += merged_entities_count
        
        return processed_count
        
    def __create_alias_relations(self, primary_node_id, alias_node_id, source_document):
        """
        创建两个节点之间的双向IS_ALIAS关系，使用MERGE避免重复创建
        
        Args:
            primary_node_id: 主节点ID
            alias_node_id: 别名节点ID
            source_document: 源文档信息（字典或字符串）
        """
        # 确保source_document是标准的字典格式
        if not isinstance(source_document, dict):
            source_document = {"filename": str(source_document), "title": "", "paragraph_number": None}
            
        # 将字典转换为JSON字符串，符合Neo4j关系属性类型要求
        source_document_json = json.dumps(source_document)
            
        with self.driver.session() as session:
            try:
                # 使用MERGE创建从别名到主节点的IS_ALIAS关系，避免重复
                alias_to_primary_query = """
                MATCH (alias)
                WHERE alias.id = $alias_node_id
                MATCH (primary)
                WHERE primary.id = $primary_node_id
                MERGE (alias)-[r:IS_ALIAS]->(primary)
                ON CREATE SET r.source_document = $source_document, r.created_at = datetime()
                ON MATCH SET r.source_document = $source_document
                RETURN elementId(r) as relation_id
                """
                
                # 使用MERGE创建从主节点到别名的IS_ALIAS关系，避免重复
                primary_to_alias_query = """
                MATCH (primary)
                WHERE primary.id = $primary_node_id
                MATCH (alias)
                WHERE alias.id = $alias_node_id
                MERGE (primary)-[r:IS_ALIAS]->(alias)
                ON CREATE SET r.source_document = $source_document, r.created_at = datetime()
                ON MATCH SET r.source_document = $source_document
                RETURN elementId(r) as relation_id
                """
                
                # 执行第一个关系创建/合并
                result1 = session.run(alias_to_primary_query, 
                                     alias_node_id=alias_node_id, 
                                     primary_node_id=primary_node_id, 
                                     source_document=source_document_json)
                rel1_record = result1.single()
                
                # 执行第二个关系创建/合并
                result2 = session.run(primary_to_alias_query, 
                                     alias_node_id=alias_node_id, 
                                     primary_node_id=primary_node_id, 
                                     source_document=source_document_json)
                rel2_record = result2.single()
                
                if rel1_record and rel2_record:
                    logger.info(f"已成功创建/合并双向IS_ALIAS关系: {alias_node_id} <-> {primary_node_id}")
                    logger.info(f"关系ID: {rel1_record['relation_id']} 和 {rel2_record['relation_id']}")
                else:
                    logger.error(f"创建/合并IS_ALIAS关系失败: {alias_node_id} <-> {primary_node_id}")
                    
            except Exception as e:
                logger.error(f"创建/合并IS_ALIAS关系时出错: {str(e)}")
                logger.error(f"主节点ID: {primary_node_id}, 别名节点ID: {alias_node_id}")
                raise
    

    
    def insert_relations(self, relations):
        """插入关系到知识图谱，只按节点名称精确匹配，并在关系存在时合并source_document列表
        
        Args:
            relations (list): 关系列表，每个关系为字典，包含以下字段：
                            - source: 源节点名称（精确匹配）
                            - target: 目标节点名称（精确匹配）
                            - type: 关系类型（默认为'RELATED_TO'）
                            - source_document: 源文档信息（可选，字典或字典列表，包含filename、title和paragraph_number字段）
            
        Returns:
            int: 成功处理的关系数量
        """
        processed_count = 0
        new_relations_count = 0  # 新增关系计数
        merged_relations_count = 0  # 合并关系计数
        
        # 创建数据库会话
        with self.driver.session() as session:
            # 处理每个关系
            for relation in relations:
                try:
                    # 验证必要字段
                    if 'source' not in relation or 'target' not in relation:
                        logger.warning(f"关系缺少必要字段: {relation}")
                        continue
                    
                    # 获取关系类型，设置默认值
                    rel_type = relation.get('type', 'RELATED_TO')
                    
                    # 使用私有方法处理source_document字段
                    source_documents = self.__prepare_source_documents(relation)
                    
                    # 精确查找源节点和目标节点 - 只按名称精确匹配
                    exact_node_query = """
                    MATCH (n)
                    WHERE (n:Concept OR n:Entity) AND n.name = $name
                    RETURN n.id AS node_id, n.name AS name, labels(n)[0] AS type
                    LIMIT 1
                    """
                    
                    # 查询源节点（精确匹配）
                    source_result = session.run(exact_node_query, name=relation['source'])
                    source_record = source_result.single()
                    
                    if not source_record:
                        logger.warning(f"找不到名称为'{relation['source']}'的节点")
                        continue
                    
                    # 查询目标节点（精确匹配）
                    target_result = session.run(exact_node_query, name=relation['target'])
                    target_record = target_result.single()
                    
                    if not target_record:
                        logger.warning(f"找不到名称为'{relation['target']}'的节点")
                        continue
                    
                    source_id = source_record['node_id']
                    source_type = source_record['type']
                    actual_source_name = source_record['name']
                    
                    target_id = target_record['node_id']
                    target_type = target_record['type']
                    actual_target_name = target_record['name']
                    
                    logger.info(f"找到关系对应的节点: {actual_source_name}({source_type}) -> {actual_target_name}({target_type})")
                    logger.info(f"准备创建关系，关系类型: {rel_type}")
                    
                    # 检查是否已存在相同类型的关系，同时获取现有关系的source_documents
                    # 由于Neo4j不支持在关系类型位置使用参数，我们需要动态构建查询
                    check_relation_query = f"""
                    MATCH (source)-[r:{rel_type}]->(target)
                    WHERE source.id = $source_id AND target.id = $target_id
                    RETURN elementId(r) AS id, r.source_documents AS existing_docs
                    LIMIT 1
                    """
                    
                    logger.info(f"准备执行关系检查查询，关系类型: {rel_type}")
                    logger.info(f"源节点ID: {source_id}, 目标节点ID: {target_id}")
                    existing_rel = session.run(check_relation_query, 
                                             source_id=source_id, 
                                             target_id=target_id).single()
                    
                    logger.info(f"关系检查结果: {existing_rel}")
                    
                    if existing_rel:
                        # 相同类型的关系已存在
                        relation_id = existing_rel['id']
                        existing_docs = existing_rel['existing_docs'] or []
                        
                        logger.info(f"现有关系ID: {relation_id}, 现有文档: {existing_docs}, 类型: {type(existing_docs)}")
                        
                        # 统一格式：将existing_docs转换为字典列表
                        normalized_existing_docs = []
                        
                        # 处理existing_docs的各种可能格式
                        if isinstance(existing_docs, str):
                            # 如果是字符串，尝试解析为JSON
                            try:
                                parsed = json.loads(existing_docs)
                                if isinstance(parsed, str):
                                    # 如果解析后还是字符串，再解析一次
                                    normalized_existing_docs = [json.loads(parsed)]
                                elif isinstance(parsed, list):
                                    # 如果是列表，处理每个元素
                                    normalized_existing_docs = self._normalize_document_list(parsed)
                                elif isinstance(parsed, dict):
                                    # 如果是字典，直接添加
                                    normalized_existing_docs = [parsed]
                            except json.JSONDecodeError:
                                logger.warning(f"无法解析JSON字符串: {existing_docs}")
                                normalized_existing_docs = []
                        elif isinstance(existing_docs, list):
                            # 如果是列表，标准化每个元素
                            normalized_existing_docs = self._normalize_document_list(existing_docs)
                        else:
                            # 其他情况设为空列表
                            normalized_existing_docs = []
                        
                        # 合并source_documents列表，按文件名去重
                        if source_documents:
                            # 合并并按文件名、标题、段落编号组合去重
                            merged_docs = normalized_existing_docs.copy()
                            # 创建现有文档的标识集合，包含文件名、标题和段落编号的组合
                            existing_doc_identifiers = set()
                            for existing_doc in merged_docs:
                                # 获取文档的标识信息，缺少的字段使用None
                                filename = existing_doc.get('filename', None)
                                title = existing_doc.get('title', None)
                                paragraph_number = existing_doc.get('paragraph_number', None)
                                # 创建唯一标识元组
                                doc_identifier = (filename, title, paragraph_number)
                                existing_doc_identifiers.add(doc_identifier)
                            
                            # 添加不存在的文档
                            for doc in source_documents:
                                # 获取当前文档的标识信息
                                filename = doc.get('filename', None)
                                title = doc.get('title', None)
                                paragraph_number = doc.get('paragraph_number', None)
                                doc_identifier = (filename, title, paragraph_number)
                                
                                # 如果标识不存在，则添加文档
                                if doc_identifier not in existing_doc_identifiers:
                                    merged_docs.append(doc)
                                    existing_doc_identifiers.add(doc_identifier)
                            
                            # 更新关系的source_documents
                            # 将merged_docs字典列表转换为JSON字符串列表，符合Neo4j关系属性类型要求
                            merged_docs_json = [json.dumps(doc) for doc in merged_docs]
                            update_query = """
                            MATCH ()-[r]->()
                            WHERE elementId(r) = $relation_id
                            SET r.source_documents = $merged_docs, r.last_updated = datetime()
                            """
                            
                            session.run(update_query, relation_id=relation_id, merged_docs=merged_docs_json)
                            logger.info(f"已更新关系的源文档列表: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type}) (ID: {relation_id})")
                            merged_relations_count += 1
                        else:
                            logger.info(f"关系已存在: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type})")
                    else:
                        # 创建新关系，包含source_documents字段
                        # 由于Neo4j不支持在关系类型位置使用参数，我们需要动态构建查询
                        create_relation_query = f"""
                        MATCH (source)
                        WHERE source.id = $source_id
                        MATCH (target)
                        WHERE target.id = $target_id
                        CREATE (source)-[r:{rel_type} {{created_at: datetime()}}]->(target)
                        """
                        
                        # 如果有source_documents，添加到关系中
                        if source_documents:
                            # 将source_documents字典列表转换为JSON字符串列表，符合Neo4j关系属性类型要求
                            source_docs_json = [json.dumps(doc) for doc in source_documents]
                            create_relation_query += "\nSET r.source_documents = $source_documents"
                            params = {'source_id': source_id, 'target_id': target_id, 'source_documents': source_docs_json}
                            logger.info(f"创建关系时包含source_documents: {len(source_documents)}个文档")
                            for i, doc in enumerate(source_documents):
                                logger.info(f"文档{i+1}: {doc.get('filename', '未知')}, 标题: {doc.get('title', '无')}, 段落: {doc.get('paragraph_number', '无')}")
                        else:
                            params = {'source_id': source_id, 'target_id': target_id}
                            logger.info("创建关系时不包含source_documents")
                        
                        create_relation_query += "\nRETURN elementId(r) AS relation_id"
                        logger.info(f"准备执行创建关系查询，关系类型: {rel_type}")
                        result = session.run(create_relation_query, **params)
                        
                        new_rel_record = result.single()
                        if new_rel_record:
                            new_rel_id = new_rel_record['relation_id']
                            logger.info(f"已创建新关系: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type}) (ID: {new_rel_id})")
                            new_relations_count += 1
                    
                    processed_count += 1
                    
                except Exception as e:
                    logger.error(f"处理关系时出错: {str(relation)} - {str(e)}")
                    continue
        
        logger.info(f"关系处理完成: 共处理 {processed_count}/{len(relations)} 个关系，其中新增 {new_relations_count} 个，更新 {merged_relations_count} 个")
        
        # 更新全局统计变量
        self.stats['relations']['new_relations'] += new_relations_count
        self.stats['relations']['merged_relations'] += merged_relations_count
        
        return processed_count
    
    # 关系处理辅助方法已移除
    # 现在关系创建使用直接的精确匹配逻辑，不再需要复杂的相似度计算和权重合并

    def get_node_statistics(self):
        """获取节点统计信息
        
        Returns:
            dict: 包含Concept和Entity类型节点数量的字典
        """
        query = """
        // 获取Concept节点数量
        MATCH (c:Concept)
        RETURN count(c) AS count, 'concept' AS type
        
        UNION
        
        // 获取Entity节点数量
        MATCH (e:Entity)
        RETURN count(e) AS count, 'entity' AS type
        """
        
        with self.driver.session() as session:
            results = session.run(query)
            stats = {"concept_count": 0, "entity_count": 0}
            
            for record in results:
                if record["type"] == "concept":
                    stats["concept_count"] = record["count"]
                elif record["type"] == "entity":
                    stats["entity_count"] = record["count"]
            
            stats["total_count"] = stats["concept_count"] + stats["entity_count"]
            return stats
    
    def _normalize_document_list(self, doc_list):
        """
        将文档列表标准化为字典列表
        
        Args:
            doc_list: 可能包含字符串或字典的文档列表
            
        Returns:
            list: 标准化的字典列表
        """
        normalized_docs = []
        for doc in doc_list:
            if isinstance(doc, str):
                try:
                    normalized_doc = json.loads(doc)
                    if isinstance(normalized_doc, dict):
                        normalized_docs.append(normalized_doc)
                    else:
                        logger.warning(f"文档解析后不是字典格式: {normalized_doc}")
                except json.JSONDecodeError:
                    logger.warning(f"无法解析JSON字符串: {doc}")
            elif isinstance(doc, dict):
                normalized_docs.append(doc)
            else:
                logger.warning(f"文档类型不支持: {type(doc)}, 值: {doc}")
        return normalized_docs

    def __prepare_source_documents(self, item):
        """
        准备源文档列表，确保是包含文件名、标题和段落编号的字典列表
        
        Args:
            item: 包含source_document字段的字典
            
        Returns:
            list: 标准化的源文档字典列表
        """
        source_documents = []
        if "source_document" in item:
            source_doc = item["source_document"]
            # 如果source_doc已经是字典格式，直接添加到列表
            if isinstance(source_doc, dict):
                source_documents = [source_doc]
            # 如果是列表，确保每个元素都是字典格式
            elif isinstance(source_doc, list):
                for doc in source_doc:
                    if isinstance(doc, dict):
                        source_documents.append(doc)
                    else:
                        # 转换为标准字典格式
                        source_documents.append({"filename": str(doc), "title": "", "paragraph_number": None})
            # 其他情况转换为标准字典格式
            else:
                source_documents = [{"filename": str(source_doc), "title": "", "paragraph_number": None}]
        return source_documents
        
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
        logger.info("数据库连接已关闭")
    
    def search_related_concepts(self, query_vector, top_k=5, threshold=0.6):
        """
        搜索与查询向量相关的概念
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最大结果数
            threshold: 相似度阈值
            
        Returns:
            相关概念列表
        """
        with self.driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('concept_vector_index', $top_k, $vector)
            YIELD node AS n, score AS similarity
            WHERE similarity > $threshold
            RETURN n.id AS id, n.name AS name, n.description AS description, 
                   n.canonicalName AS canonical_name, n.sourceDocuments AS sourceDocuments, similarity
            ORDER BY similarity DESC
            """
            results = list(session.run(query, vector=query_vector.tolist(), 
                                      top_k=top_k, threshold=threshold))
            
            return [{
                'id': record['id'],
                'name': record['name'],
                'description': record['description'],
                'canonical_name': record['canonical_name'],
                'source_documents': record['sourceDocuments'],
                'similarity': record['similarity']
            } for record in results]
    
    def search_related_entities(self, query_vector, top_k=5, threshold=0.6):
        """
        搜索与查询向量相关的实体
        
        Args:
            query_vector: 查询向量
            top_k: 返回的最大结果数
            threshold: 相似度阈值
            
        Returns:
            相关实体列表
        """
        with self.driver.session() as session:
            query = """
            CALL db.index.vector.queryNodes('entity_vector_index', $top_k, $vector)
            YIELD node AS n, score AS similarity
            WHERE similarity > $threshold
            RETURN n.id AS id, n.name AS name, n.description AS description, 
                   n.canonicalName AS canonical_name, n.sourceDocuments AS sourceDocuments, similarity
            ORDER BY similarity DESC
            """
            results = list(session.run(query, vector=query_vector.tolist(), 
                                      top_k=top_k, threshold=threshold))
            
            return [{
                'id': record['id'],
                'name': record['name'],
                'description': record['description'],
                'canonical_name': record['canonical_name'],
                'source_documents': record['sourceDocuments'],
                'similarity': record['similarity']
            } for record in results]
    
    def get_node_relations(self, node_id, node_type, limit=5):
        """
        获取节点的相关关系
        
        Args:
            node_id: 节点ID
            node_type: 节点类型（Concept或Entity）
            limit: 返回的最大关系数
            
        Returns:
            关系列表
        """
        with self.driver.session() as session:
            # 先构建带有正确节点标签的查询字符串
            query = f"""
            MATCH (n:{node_type}) 
            WHERE n.id = $node_id
            MATCH (n)-[r]-(related)
            RETURN type(r) AS relation_type, 
                   related.name AS related_name,
                   labels(related)[0] AS related_node_type
            LIMIT $limit
            """
            results = list(session.run(query, node_id=node_id, limit=limit))
            
            return [{
                'relation_type': record['relation_type'],
                'related_name': record['related_name'],
                'related_node_type': record['related_node_type']
            } for record in results]
    
    def search_knowledge_by_text(self, query_text, top_k=5, max_hops=2, dot_threshold=0.95):
        """
        基于文本搜索相关知识，并在找到最近节点后搜索相似节点
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数
            max_hops: 搜索相似节点的最大跳数
            dot_threshold: 向量相似度的点积阈值
            
        Returns:
            组合的相关知识列表，包含相似节点信息
        """
        import time
        start_time = time.time()
        
        # 记录查询开始
        query_length = len(query_text)
        query_preview = query_text[:100] + "..." if query_length > 100 else query_text
        logger.info(f"[SENTENCE_TRANSFORMER] search_knowledge_by_text开始 - 查询长度: {query_length}, 预览: '{query_preview}', top_k: {top_k}")
        
        try:
            # 获取查询向量
            encode_start = time.time()
            query_vector = self.model.encode(query_text)
            encode_time = time.time() - encode_start
            logger.info(f"[SENTENCE_TRANSFORMER] 查询向量编码完成 - 耗时: {encode_time:.3f}s, 向量维度: {query_vector.shape if hasattr(query_vector, 'shape') else len(query_vector)}")
            
            # 搜索相关概念和实体
            search_start = time.time()
            concepts = self.search_related_concepts(query_vector, top_k)
            entities = self.search_related_entities(query_vector, top_k)
            search_time = time.time() - search_start
            logger.info(f"[SENTENCE_TRANSFORMER] 概念和实体搜索完成 - 耗时: {search_time:.3f}s, 找到概念: {len(concepts)}, 实体: {len(entities)}")
            
            # 合并结果并按相似度排序
            all_results = []
            for concept in concepts:
                concept['node_type'] = 'Concept'
                all_results.append(concept)
            
            for entity in entities:
                entity['node_type'] = 'Entity'
                all_results.append(entity)
            
            # 按相似度排序
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            
            # 为每个结果添加关系信息和相似节点
            for i, result in enumerate(all_results[:top_k]):
                logger.info(f"[SENTENCE_TRANSFORMER] 处理第{i+1}/{min(top_k, len(all_results))}个结果 - 节点: {result.get('name', 'Unknown')}")
                
                relations_start = time.time()
                relations = self.get_node_relations(result['id'], result['node_type'])
                relations_time = time.time() - relations_start
                logger.info(f"[SENTENCE_TRANSFORMER] 获取关系信息完成 - 耗时: {relations_time:.3f}s, 关系数: {len(relations)}")
                result['relations'] = relations
                
                # 调用search_similar_nodes寻找其他相似节点
                similar_start = time.time()
                similar_nodes = self.search_similar_nodes(
                    start_node_id=result['id'],
                    start_node_type=result['node_type'],
                    max_hops=max_hops,
                    dot_threshold=dot_threshold
                )
                similar_time = time.time() - similar_start
                logger.info(f"[SENTENCE_TRANSFORMER] 搜索相似节点完成 - 耗时: {similar_time:.3f}s, 相似节点数: {len(similar_nodes)}")
                result['similar_nodes'] = similar_nodes
            
            total_time = time.time() - start_time
            logger.info(f"[SENTENCE_TRANSFORMER] search_knowledge_by_text完成 - 总耗时: {total_time:.3f}s, 返回结果数: {len(all_results[:top_k])}")
            
            return all_results[:top_k]
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"[SENTENCE_TRANSFORMER] search_knowledge_by_text出错 - 耗时: {error_time:.3f}s, 查询: '{query_text[:50]}...', 错误: {str(e)}")
            return []
    
    def _find_adjacent_nodes(self, node_id, node_type, dot_threshold, session):
        """
        查找指定节点的所有相邻节点
        
        Args:
            node_id (str): 节点ID
            node_type (str): 节点类型
            dot_threshold (float): 向量相似度阈值
            session: Neo4j会话
            
        Returns:
            list: 相邻节点列表
        """
        logger.info(f"[SENTENCE_TRANSFORMER] 查找节点 {node_id[:8]}... 的相邻节点")
        
        try:
            # 1. 尝试使用APOC插件查找向量相似节点，如果失败则使用替代方案
            try:
                similar_nodes_query = f"""
                MATCH (start:{node_type} {{id: $node_id}})
                MATCH (other:{node_type})
                WHERE other.id <> start.id 
                  AND other.embedding IS NOT NULL
                  AND start.embedding IS NOT NULL
                WITH start, other, 
                     apoc.math.cosineSimilarity(start.embedding, other.embedding) AS similarity
                WHERE similarity > $dot_threshold
                RETURN other.id AS id, other.name AS name, similarity
                ORDER BY similarity DESC
                """
                
                similar_results = list(session.run(similar_nodes_query, 
                                                  node_id=node_id, 
                                                  dot_threshold=dot_threshold))
                logger.info(f"[SENTENCE_TRANSFORMER] 使用APOC插件找到 {len(similar_results)} 个相似节点")
                
            except Exception as apoc_error:
                logger.warning(f"[SENTENCE_TRANSFORMER] APOC插件不可用，使用替代方案: {str(apoc_error)}")
                
                # 使用reduce函数计算余弦相似度的替代方案
                similar_nodes_query = f"""
                MATCH (start:{node_type} {{id: $node_id}})
                MATCH (other:{node_type})
                WHERE other.id <> start.id 
                  AND other.embedding IS NOT NULL
                  AND start.embedding IS NOT NULL
                WITH start, other, start.embedding AS start_vec, other.embedding AS other_vec
                WITH start, other,
                     reduce(dot = 0.0, i IN range(0, size(start_vec)-1) | dot + start_vec[i] * other_vec[i]) AS dot_product,
                     reduce(sq1 = 0.0, i IN range(0, size(start_vec)-1) | sq1 + start_vec[i] * start_vec[i]) AS start_norm_sq,
                     reduce(sq2 = 0.0, i IN range(0, size(other_vec)-1) | sq2 + other_vec[i] * other_vec[i]) AS other_norm_sq
                WITH start, other, 
                     CASE WHEN start_norm_sq > 0 AND other_norm_sq > 0
                          THEN dot_product / (sqrt(start_norm_sq) * sqrt(other_norm_sq))
                          ELSE 0.0
                     END AS similarity
                WHERE similarity > $dot_threshold
                RETURN other.id AS id, other.name AS name, similarity
                ORDER BY similarity DESC
                """
                
                similar_results = list(session.run(similar_nodes_query, 
                                                  node_id=node_id, 
                                                  dot_threshold=dot_threshold))
                logger.info(f"[SENTENCE_TRANSFORMER] 使用替代方案找到 {len(similar_results)} 个相似节点")
            
            # 2. 查找IS_ALIAS连接的节点
            alias_query = f"""
            MATCH (start:{node_type} {{id: $node_id}})-[:IS_ALIAS]-(alias:{node_type})
            RETURN alias.id AS id, alias.name AS name, 1.0 AS similarity
            """
            
            alias_results = list(session.run(alias_query, node_id=node_id))
            logger.info(f"[SENTENCE_TRANSFORMER] 找到 {len(alias_results)} 个IS_ALIAS节点")
            
            # 3. 合并结果
            adjacent_nodes = []
            for record in similar_results + alias_results:
                adjacent_nodes.append({
                    'id': record['id'],
                    'name': record['name'],
                    'similarity': record['similarity']
                })
            
            logger.info(f"[SENTENCE_TRANSFORMER] 节点 {node_id[:8]}... 总共找到 {len(adjacent_nodes)} 个相邻节点")
            return adjacent_nodes
            
        except Exception as e:
            logger.error(f"[SENTENCE_TRANSFORMER] 查找相邻节点时发生错误: {str(e)}")
            return []

    def search_similar_nodes(self, start_node_id, start_node_type, max_hops, dot_threshold):
        """
        从指定节点开始，使用BFS搜索所有至多max_hops跳相邻的同类型节点列表
        
        相邻定义：
        1. 通过IS_ALIAS直接连接的同类节点（1跳）
        2. 向量相似度大于dot_threshold的同类节点（1跳）
        
        Args:
            start_node_id (str): 起始节点ID
            start_node_type (str): 起始节点类型（'Concept' 或 'Entity'）
            max_hops (int): 最大跳数限制
            dot_threshold (float): 向量相似度的点积阈值
            
        Returns:
            list: 相邻节点列表
                [
                    {
                        'id': str,
                        'name': str,
                        'type': str,
                        'hops': int,  # 距离起始节点的跳数
                        'path': list  # 从起始节点到此节点的路径
                    }
                ]
        """
        results = []
        
        logger.info(f"[SENTENCE_TRANSFORMER] 开始BFS搜索相似节点 - 起始节点: {start_node_type}({start_node_id}), 最大跳数: {max_hops}, 点积阈值: {dot_threshold}")
        
        with self.driver.session() as session:
            # 1. 验证起始节点存在
            start_node_query = f"""
            MATCH (n:{start_node_type})
            WHERE n.id = $start_node_id
            RETURN n.id AS id, n.name AS name
            """
            
            start_node_result = list(session.run(start_node_query, start_node_id=start_node_id))
            if not start_node_result:
                logger.warning(f"未找到节点: {start_node_type}({start_node_id})")
                return results
            
            start_node = start_node_result[0]
            
            # 2. BFS搜索
            visited = set()
            queue = [(start_node_id, 0, [start_node['name']])]  # (node_id, hops, path)
            
            while queue and len(results) < 100:  # 限制结果数量避免过多
                current_id, hops, path = queue.pop(0)
                
                if current_id in visited or hops > max_hops:
                    continue
                
                visited.add(current_id)
                logger.info(f"[SENTENCE_TRANSFORMER] BFS处理节点: {current_id[:8]}..., 跳数: {hops}, 已访问: {len(visited)}")
                
                # 跳过起始节点本身
                if hops > 0:
                    # 获取当前节点信息
                    current_node_query = f"""
                    MATCH (n:{start_node_type})
                    WHERE n.id = $node_id
                    RETURN n.id AS id, n.name AS name
                    """
                    
                    current_node_result = list(session.run(current_node_query, node_id=current_id))
                    if current_node_result:
                        current_node = current_node_result[0]
                        results.append({
                            'id': current_node['id'],
                            'name': current_node['name'],
                            'type': start_node_type,
                            'hops': hops,
                            'path': path.copy()
                        })
                
                # 如果还没达到最大跳数，查找相邻节点并加入队列
                if hops < max_hops:
                    try:
                        adjacent_nodes = self._find_adjacent_nodes(current_id, start_node_type, dot_threshold, session)
                        
                        for neighbor in adjacent_nodes:
                            if neighbor['id'] not in visited:
                                new_path = path.copy()
                                new_path.append(neighbor['name'])
                                queue.append((neighbor['id'], hops + 1, new_path))
                                
                    except Exception as e:
                        logger.error(f"[SENTENCE_TRANSFORMER] 查找节点 {current_id[:8]}... 的相邻节点时出错: {str(e)}")
                        continue
            
            # 3. 按跳数和名称排序
            results.sort(key=lambda x: (x['hops'], x['name']))
            
            logger.info(f"[SENTENCE_TRANSFORMER] BFS搜索完成 - 从节点 {start_node['name']}({start_node_id[:8]}...) 找到 {len(results)} 个相邻的{start_node_type}节点")
            
            return results
