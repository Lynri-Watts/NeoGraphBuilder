from neo4j import GraphDatabase
import numpy as np
from sentence_transformers import SentenceTransformer
import spacy
import os
import time
import logging

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
            
            session.run("""
            CREATE TEXT INDEX concept_canonical_name_index IF NOT EXISTS 
            FOR (c:Concept) ON (c.canonicalName)
            """)
            
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
            
            session.run("""
            CREATE TEXT INDEX entity_canonical_name_index IF NOT EXISTS 
            FOR (e:Entity) ON (e.canonicalName)
            """)
            
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
        # 确保文本是字符串类型
        if not isinstance(text, str) or not text.strip():
            logger.warning(f"无效文本输入: {text}")
            # 返回一个零向量作为默认值
            return [0.0] * self.model.get_sentence_embedding_dimension()
            
        try:
            # 逐一生成向量，确保更准确的编码
            vector = self.model.encode(text, convert_to_numpy=True)
            
            # 验证向量是否有效
            if vector is None or len(vector) == 0:
                logger.warning(f"生成的向量为空: {text}")
                return [0.0] * self.model.get_sentence_embedding_dimension()
                
            # 检查向量中是否有NaN值
            if np.isnan(vector).any():
                logger.warning(f"生成的向量包含NaN值: {text}")
                return [0.0] * self.model.get_sentence_embedding_dimension()
                
            return vector.tolist()
            
        except Exception as e:
            logger.error(f"生成向量时出错: {text} - {str(e)}")
            return [0.0] * self.model.get_sentence_embedding_dimension()
    
    def __find_similar_nodes(self, node):
        """在Neo4j中查找相似节点，确保Concept和Entity永远不相似"""
        node_type = node.get("type", "Concept")
        
        # 根据节点类型设置不同的索引和查询参数
        if node_type == "Concept":
            label = "Concept"
            vector_index = "concept_vector_index"
            # 概念相似度阈值较高，因为概念更抽象
            threshold = 0.75
        else:  # Entity
            label = "Entity"
            vector_index = "entity_vector_index"
            # 实体相似度阈值可以稍低，因为实体更具体
            threshold = 0.7
        
        # 分别执行不同的匹配查询，以便统计每种方法的贡献
        with self.driver.session() as session:
            # 1. 精确匹配查询
            exact_match_query = f"""
            MATCH (n:{label})
            WHERE n.canonicalName = $canonical_name 
            RETURN n.id AS node_id, n.name AS name, 'exact_match' AS match_type
            LIMIT 1
            """
            exact_results = list(session.run(exact_match_query, 
                                           canonical_name=node["canonical_name"]))
            
            if exact_results:
                self.stats['similarity_matches']['exact_match'] += 1
                return [{"concept_id": record["node_id"], "name": record["name"], "similarity": 1.0, "match_type": "exact_match"} 
                        for record in exact_results]
            
            # 2. 别名匹配查询
            alias_match_query = f"""
            MATCH (n:{label})
            WHERE $name IN n.aliases
            RETURN n.id AS node_id, n.name AS name, 'alias_match' AS match_type
            LIMIT 1
            """
            alias_results = list(session.run(alias_match_query, name=node["name"]))
            
            if alias_results:
                self.stats['similarity_matches']['alias_match'] += 1
                return [{"concept_id": record["node_id"], "name": record["name"], "similarity": 0.95, "match_type": "alias_match"} 
                        for record in alias_results]
            
            # 3. 模糊匹配查询
            fuzzy_match_query = f"""
            MATCH (n:{label})
            WHERE n.canonicalName CONTAINS $canonical_name 
               OR ANY(alias IN n.aliases WHERE alias CONTAINS $name)
            RETURN n.id AS node_id, n.name AS name, 'fuzzy_match' AS match_type
            LIMIT 5
            """
            fuzzy_results = list(session.run(fuzzy_match_query, 
                                           name=node["name"],
                                           canonical_name=node["canonical_name"]))
            
            if fuzzy_results:
                self.stats['similarity_matches']['fuzzy_match'] += 1
                return [{"concept_id": record["node_id"], "name": record["name"], "similarity": 0.8, "match_type": "fuzzy_match"} 
                        for record in fuzzy_results]
            
            # 4. 向量相似度搜索
            vector_query = f"""
            CALL db.index.vector.queryNodes('{vector_index}', 10, $vector)
            YIELD node AS n, score AS similarity
            WHERE similarity > $threshold
            RETURN n.id AS node_id, n.name AS name, similarity, 'vector_match' AS match_type
            """
            vector_results = list(session.run(vector_query, 
                                           vector=node["vector"],
                                           threshold=threshold))
            
            if vector_results:
                self.stats['similarity_matches']['vector_match'] += 1
                return [{"concept_id": record["node_id"], "name": record["name"], "similarity": record["similarity"], "match_type": "vector_match"} 
                        for record in vector_results]
            
            # 没有找到相似节点
            return []
    
    def __merge_or_create_node(self, node, candidates):
        """根据相似度决定合并或创建新节点，对Concept和Entity使用不同的判断逻辑"""
        node_type = node.get("type", "Concept")
        
        # 如果没有候选，直接创建新节点
        if not candidates:
            return self.__create_new_node(node)
        
        # 选择最佳候选
        best_candidate = max(candidates, key=lambda x: x["similarity"])
        
        # 根据节点类型使用不同的相似度阈值
        if node_type == "Concept":
            # 概念相似度要求更高
            if best_candidate["similarity"] > 0.95:
                self.stats['nodes']['merged_concepts'] += 1
                return self.__merge_node(best_candidate["concept_id"], node)
            
            # 概念更注重上下文和语义关联
            context_similarity = self.__check_context_similarity(
                best_candidate["concept_id"], 
                node["context"]
            )
            
            # 概念的上下文权重更高
            combined_similarity = 0.6 * best_candidate["similarity"] + 0.4 * context_similarity
            
            if combined_similarity > 0.92:
                self.stats['nodes']['merged_concepts'] += 1
                if context_similarity > 0.5:  # 如果上下文相似度较高，则记录上下文匹配
                    self.stats['similarity_matches']['context_match'] += 1
                return self.__merge_node(best_candidate["concept_id"], node)
        else:  # Entity
            # 实体更注重名称匹配
            if best_candidate["similarity"] > 0.9:
                self.stats['nodes']['merged_entities'] += 1
                return self.__merge_node(best_candidate["concept_id"], node)
            
            # 实体的上下文权重较低
            context_similarity = self.__check_context_similarity(
                best_candidate["concept_id"], 
                node["context"]
            )
            
            combined_similarity = 0.8 * best_candidate["similarity"] + 0.2 * context_similarity
            
            if combined_similarity > 0.88:
                self.stats['nodes']['merged_entities'] += 1
                if context_similarity > 0.5:  # 如果上下文相似度较高，则记录上下文匹配
                    self.stats['similarity_matches']['context_match'] += 1
                return self.__merge_node(best_candidate["concept_id"], node)
        
        # 不符合合并条件，创建新节点
        if node_type == "Concept":
            self.stats['nodes']['new_concepts'] += 1
        else:
            self.stats['nodes']['new_entities'] += 1
        return self.__create_new_node(node)
    
    def __check_context_similarity(self, node_id, context):
        """检查节点上下文相似度，同时查询Concept和Entity类型"""
        query = """
        // 同时查询Concept和Entity类型的节点
        MATCH (n) 
        WHERE n.id = $node_id AND (n:Concept OR n:Entity)
        MATCH (n)-[:RELATED_TO]->(other)
        WITH collect(other.name) AS related_nodes
        
        RETURN CASE WHEN size(related_nodes) > 0 
                   THEN size([name IN related_nodes WHERE name IN $context]) * 1.0 / size(related_nodes) 
                   ELSE 0 END AS context_similarity
        """
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id, context=context)
            record = result.single()
            return record["context_similarity"] if record else 0
    
    def __create_new_node(self, node):
        """创建新节点，根据类型创建Concept或Entity"""
        node_type = node.get("type", "Concept")
        
        # 获取节点的别名列表，如果没有提供则默认为仅包含名称的列表
        aliases = node.get("aliases", [])
        # 确保名称也在别名列表中
        if node["name"] not in aliases:
            aliases.append(node["name"])
            
        # 获取description，默认为空字符串
        description = node.get("description", "")
            
        query = f"""
        CREATE (n:{node_type} {{
            id: apoc.create.uuid(),
            name: $name,
            canonicalName: $canonical_name,
            aliases: $aliases,
            description: $description,
            embedding: $vector,
            sourceDocuments: [$source_document],
            type: $node_type
        }})
        RETURN n.id AS node_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                name=node["name"],
                canonical_name=node["canonical_name"],
                description=description,
                vector=node["vector"],
                source_document=node["source_document"],
                aliases=aliases,
                node_type=node_type
            )
            record = result.single()
            return record["node_id"]
    
    def __merge_node(self, existing_id, new_node):
        """合并到现有节点，同时合并别称列表"""
        # 获取新节点的别名列表，如果没有提供则默认为空列表
        new_aliases = new_node.get("aliases", [])
        # 确保名称也在别名列表中
        if new_node["name"] not in new_aliases:
            new_aliases.append(new_node["name"])
            
        # 获取新节点的description，默认为空字符串
        new_description = new_node.get("description", "")
        
        # 构建查询，处理description可能为空的情况
        query = """
        // 匹配Concept或Entity类型的节点
        MATCH (n) 
        WHERE n.id = $existing_id AND (n:Concept OR n:Entity)
        SET n.aliases = CASE WHEN size([alias IN $new_aliases WHERE NOT alias IN n.aliases]) > 0 
                             THEN [alias IN n.aliases] + [alias IN $new_aliases WHERE NOT alias IN n.aliases]
                             ELSE n.aliases END
        SET n.description = CASE WHEN $new_description <> '' 
                                THEN CASE WHEN n.description IS NOT NULL AND n.description <> '' 
                                          THEN n.description + '\n' + $new_description
                                          ELSE $new_description
                                     END
                                ELSE n.description END
        SET n.sourceDocuments = n.sourceDocuments + $source_document
        RETURN n.id AS node_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                existing_id=existing_id,
                new_aliases=new_aliases,
                new_description=new_description,
                source_document=new_node["source_document"]
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
                node_id = self.__merge_or_create_node(full_node, similar_nodes)
                
                # 统计新增和合并的节点，按类型分类
                if node['type'] == 'Concept':
                    if similar_nodes and any(c['similarity'] > 0.8 for c in similar_nodes):
                        merged_concepts_count += 1
                        logger.info(f"已合并概念 '{node['name']}' -> ID: {node_id}")
                    else:
                        new_concepts_count += 1
                        logger.info(f"已新增概念 '{node['name']}' -> ID: {node_id}")
                else:  # Entity
                    if similar_nodes and any(c['similarity'] > 0.8 for c in similar_nodes):
                        merged_entities_count += 1
                        logger.info(f"已合并实体 '{node['name']}' -> ID: {node_id}")
                    else:
                        new_entities_count += 1
                        logger.info(f"已新增实体 '{node['name']}' -> ID: {node_id}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理节点 '{node.get('name', 'Unknown')}' 时出错: {str(e)}")
                continue
        
        # 输出最终统计信息
        logger.info(f"节点处理完成: 总共处理 {processed_count} 个节点")
        logger.info(f"概念统计: 新增 {new_concepts_count} 个，合并 {merged_concepts_count} 个")
        logger.info(f"实体统计: 新增 {new_entities_count} 个，合并 {merged_entities_count} 个")
        
        # 更新全局统计变量
        self.stats['nodes']['new_concepts'] = new_concepts_count
        self.stats['nodes']['merged_concepts'] = merged_concepts_count
        self.stats['nodes']['new_entities'] = new_entities_count
        self.stats['nodes']['merged_entities'] = merged_entities_count
        
        return processed_count
    
    def insert_concepts(self, concepts):
        """兼容旧版本的概念插入方法，将概念转换为节点后插入
        
        Args:
            concepts (list): 概念列表，每个概念为字典
            
        Returns:
            int: 成功处理的概念数量
        """
        # 将概念转换为默认类型为Concept的节点
        nodes = []
        for concept in concepts:
            node = concept.copy()
            node['type'] = 'Concept'
            nodes.append(node)
        
        # 调用新的节点插入方法
        return self.insert_nodes(nodes)
    
    def insert_relations(self, relations):
        """插入关系到知识图谱
        
        Args:
            relations (list): 关系列表，每个关系为字典，包含以下字段：
                            - source: 源节点名称
                            - target: 目标节点名称
                            - type: 关系类型（默认为'RELATED_TO'）
                            - weight: 关系权重（默认为1.0）
                            - properties: 可选的额外属性字典
                            - similarity_threshold: 相似度阈值（默认为0.7）
            
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
                    
                    # 获取关系属性，设置默认值
                    rel_type = relation.get('type', 'RELATED_TO')
                    weight = relation.get('weight', 1.0)
                    properties = relation.get('properties', {})
                    similarity_threshold = relation.get('similarity_threshold', 0.7)
                    
                    # 查找源节点和目标节点 - 支持Concept和Entity类型
                    source_node_query = """
                    MATCH (n)
                    WHERE (n:Concept OR n:Entity) AND 
                          (n.name = $name OR $name IN n.aliases OR 
                           n.canonicalName = $canonical_name)
                    RETURN n.id AS node_id, n.name AS name, labels(n)[0] AS type,
                           n.embedding AS embedding
                    ORDER BY CASE 
                        WHEN n.name = $name THEN 3
                        WHEN $name IN n.aliases THEN 2
                        WHEN n.canonicalName = $canonical_name THEN 1
                        ELSE 0 
                    END DESC
                    LIMIT 10
                    """
                    
                    # 准备参数
                    source_canonical = self.nlp(relation['source'])[0].lemma_.lower() if self.nlp(relation['source']) else relation['source'].lower()
                    target_canonical = self.nlp(relation['target'])[0].lemma_.lower() if self.nlp(relation['target']) else relation['target'].lower()
                    
                    # 查询源节点
                    source_nodes = session.run(source_node_query, 
                                             name=relation['source'], 
                                             canonical_name=source_canonical)
                    source_nodes = list(source_nodes)
                    
                    # 查询目标节点
                    target_nodes = session.run(source_node_query, 
                                             name=relation['target'], 
                                             canonical_name=target_canonical)
                    target_nodes = list(target_nodes)
                    
                    if not source_nodes:
                        logger.warning(f"找不到与'{relation['source']}'相关的节点（Concept或Entity）")
                        continue
                    
                    if not target_nodes:
                        logger.warning(f"找不到与'{relation['target']}'相关的节点（Concept或Entity）")
                        continue
                    
                    # 选择最佳匹配节点
                    source_node = source_nodes[0]
                    target_node = target_nodes[0]
                    
                    source_id = source_node['node_id']
                    source_type = source_node['type']
                    actual_source_name = source_node['name']
                    
                    target_id = target_node['node_id']
                    target_type = target_node['type']
                    actual_target_name = target_node['name']
                    
                    logger.info(f"找到关系对应的节点: {actual_source_name}({source_type}) -> {actual_target_name}({target_type})")
                    
                    # 检查现有关系
                    existing_relations = self.__check_existing_relations(session, source_id, target_id)
                    
                    if existing_relations:
                        # 关系已存在，检查是否相似
                        relation_exists = False
                        for existing_rel in existing_relations:
                            if self.__verify_relation_similarity(existing_rel, rel_type, properties):
                                # 关系相似，合并权重
                                self.__merge_relation_weights(session, existing_rel['id'], weight, properties)
                                logger.info(f"已合并相似关系: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type}) (ID: {existing_rel['id']})")
                                relation_exists = True
                                merged_relations_count += 1
                                processed_count += 1
                                break
                        
                        if not relation_exists:
                            # 关系不相似，创建新关系
                            new_rel_id = self.__create_new_relation(session, source_id, target_id, rel_type, weight, properties)
                            logger.info(f"已创建不同类型关系: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type}) (ID: {new_rel_id})")
                            new_relations_count += 1
                            processed_count += 1
                    else:
                        # 关系不存在，创建新关系
                        new_rel_id = self.__create_new_relation(session, source_id, target_id, rel_type, weight, properties)
                        logger.info(f"已创建新关系: '{actual_source_name}'({source_type}) -[{rel_type}]-> '{actual_target_name}'({target_type}) (ID: {new_rel_id})")
                        new_relations_count += 1
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"处理关系时出错: {str(relation)} - {str(e)}")
                    continue
        
        logger.info(f"关系处理完成: 共处理 {processed_count}/{len(relations)} 个关系，其中新增 {new_relations_count} 个，合并 {merged_relations_count} 个")
        
        # 更新全局统计变量
        self.stats['relations']['new_relations'] = new_relations_count
        self.stats['relations']['merged_relations'] = merged_relations_count
        
        return processed_count
    
    def __check_existing_relations(self, session, source_id, target_id):
        """检查源节点和目标节点之间是否存在关系"""
        query = """
        MATCH (source)-[r]->(target)
        WHERE source.id = $source_id AND target.id = $target_id
        RETURN elementId(r) AS id, type(r) AS type, properties(r) AS properties
        """
        result = session.run(query, source_id=source_id, target_id=target_id)
        return [record for record in result]
    
    def __verify_relation_similarity(self, existing_rel, new_type, new_properties):
        """基于语义相似度验证关系是否相似
        
        判断标准：
        1. 关系类型基于语义相似度比较
        2. 属性基于语义相似度比较
        
        Returns:
            bool: 如果关系语义相似则返回True，否则返回False
        """
        # 设置相似度阈值
        type_similarity_threshold = 0.8
        property_similarity_threshold = 0.7
        
        # 计算关系类型的语义相似度
        existing_type = existing_rel['type']
        type_similarity = self.__calculate_text_similarity(existing_type, new_type)
        
        logger.debug(f"关系类型语义相似度: '{existing_type}' vs '{new_type}' = {type_similarity:.4f}")
        
        # 如果关系类型语义不相似，直接返回False
        if type_similarity < type_similarity_threshold:
            return False
        
        # 检查属性语义相似度
        existing_props = existing_rel['properties']
        
        # 对所有非weight和last_updated的属性进行语义比较
        for key, new_value in new_properties.items():
            if key in ['weight', 'last_updated']:
                continue
                
            if key in existing_props:
                existing_value = existing_props[key]
                # 只对字符串类型的值进行语义比较
                if isinstance(existing_value, str) and isinstance(new_value, str):
                    prop_similarity = self.__calculate_text_similarity(existing_value, new_value)
                    logger.debug(f"属性语义相似度: '{key}' = '{existing_value}' vs '{new_value}' = {prop_similarity:.4f}")
                    
                    if prop_similarity < property_similarity_threshold:
                        return False
                # 对于非字符串类型，仍然使用精确匹配
                elif existing_value != new_value:
                    logger.debug(f"非字符串属性不匹配: '{key}' = {existing_value} vs {new_value}")
                    return False
        
        logger.debug(f"关系语义相似，通过验证")
        return True
    
    def __calculate_text_similarity(self, text1, text2):
        """计算两个文本的语义相似度
        
        Args:
            text1: 第一个文本
            text2: 第二个文本
            
        Returns:
            float: 0-1之间的相似度分数
        """
        # 生成文本向量
        vec1 = self.__generate_vector(text1)
        vec2 = self.__generate_vector(text2)
        
        # 计算余弦相似度
        similarity = self.__cosine_similarity(vec1, vec2)
        
        return similarity
    
    def __cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度
        
        Args:
            vec1: 第一个向量
            vec2: 第二个向量
            
        Returns:
            float: 余弦相似度值
        """
        # 确保向量是numpy数组
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # 计算点积
        dot_product = np.dot(vec1, vec2)
        
        # 计算向量范数
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        # 防止除零错误
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 计算余弦相似度
        similarity = dot_product / (norm1 * norm2)
        
        return similarity
    
    def __merge_relation_weights(self, session, relation_id, new_weight, properties=None):
        """合并关系权重
        
        Args:
            session: Neo4j会话
            relation_id: 关系ID
            new_weight: 新权重
            properties: 要合并的额外属性
            
        Returns:
            float: 更新后的权重
        """
        # 获取当前关系权重
        query = """
        MATCH ()-[r]->()
        WHERE elementId(r) = $relation_id
        RETURN r.weight AS current_weight, properties(r) AS props
        """
        
        result = session.run(query, relation_id=relation_id)
        record = result.single()
        
        if not record:
            raise ValueError(f"关系不存在: {relation_id}")
        
        current_weight = record["current_weight"] if record["current_weight"] is not None else 0.0
        current_props = record["props"]
        
        # 计算新权重（使用加权平均）
        updated_weight = 0.7 * current_weight + 0.3 * new_weight
        
        # 更新属性
        update_props = {"weight": updated_weight, "last_updated": datetime.now().isoformat()}
        if properties:
            # 合并额外属性，但不覆盖现有属性
            for key, value in properties.items():
                if key not in update_props:
                    update_props[key] = value
        
        # 更新关系
        query = """
        MATCH ()-[r]->()
        WHERE elementId(r) = $relation_id
        SET r = r + $update_props
        """
        
        session.run(query, relation_id=relation_id, update_props=update_props)
        return updated_weight
    
    def __create_new_relation(self, session, source_id, target_id, rel_type, weight, properties):
        """创建新关系"""
        params = {
            'source_id': source_id,
            'target_id': target_id,
            'weight': weight,
            **properties
        }
        
        query = f"""
        MATCH (source)
        WHERE source.id = $source_id
        MATCH (target)
        WHERE target.id = $target_id
        CREATE (source)-[r:{rel_type} {{weight: $weight, last_updated: datetime()}}]->(target)
        """
        
        # 添加额外属性
        for key, value in properties.items():
            if key not in ['weight', 'last_updated']:
                query += f"\n        SET r.{key} = ${key}"
        
        query += "\n        RETURN elementId(r) AS relation_id"
        
        result = session.run(query, params)
        record = result.single()
        return record['relation_id'] if record else None
    
    def get_node_by_id(self, node_id):
        """通过ID获取节点信息
        
        Args:
            node_id: 节点ID
            
        Returns:
            dict: 节点信息字典
        """
        query = """
        MATCH (n) 
        WHERE n.id = $node_id AND (n:Concept OR n:Entity)
        RETURN n.id AS id, n.name AS name, n.description AS description, 
               n.canonicalName AS canonical_name, n.aliases AS aliases,
               n.embedding AS embedding, n.sourceDocuments AS source_documents,
               labels(n)[0] AS type
        """
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()
            
            if not record:
                return None
            
            return {
                "id": record["id"],
                "name": record["name"],
                "description": record["description"],
                "canonical_name": record["canonical_name"],
                "aliases": record["aliases"],
                "embedding": record["embedding"],
                "source_documents": record["source_documents"],
                "type": record["type"]
            }
    
    def get_concept_by_id(self, concept_id):
        """通过ID获取概念信息（兼容旧版方法）
        
        Args:
            concept_id: 概念ID
            
        Returns:
            dict: 概念信息字典
        """
        node = self.get_node_by_id(concept_id)
        # 如果找到节点且类型为Concept，返回节点信息
        if node and node.get('type') == 'Concept':
            # 移除type字段以保持向后兼容
            node_copy = node.copy()
            node_copy.pop('type', None)
            return node_copy
        return None
        
    def get_nodes_by_name(self, name, node_type=None, limit=10):
        """通过名称查找节点
        
        Args:
            name: 节点名称
            node_type: 节点类型 ('Concept' 或 'Entity'，None表示两种类型都搜索)
            limit: 返回结果数量限制
            
        Returns:
            list: 节点列表
        """
        # 根据node_type构建节点类型条件
        type_condition = "" if node_type is None else f"AND n:{node_type}"
        
        query = f"""
        // 1. 精确匹配名称
        MATCH (n) 
        WHERE (n:Concept OR n:Entity) {type_condition} AND
              (n.name = $name OR $name IN n.aliases)
        RETURN n.id AS id, n.name AS name, n.description AS description, 
               n.canonicalName AS canonical_name, n.aliases AS aliases,
               n.embedding AS embedding, n.sourceDocuments AS source_documents,
               labels(n)[0] AS type,
               1.0 AS match_score
        LIMIT $limit
        
        UNION
        
        // 2. 模糊匹配名称
        MATCH (n) 
        WHERE (n:Concept OR n:Entity) {type_condition} AND
              (n.name CONTAINS $name OR n.canonicalName CONTAINS $name
               OR ANY(alias IN n.aliases WHERE alias CONTAINS $name))
        RETURN n.id AS id, n.name AS name, n.description AS description, 
               n.canonicalName AS canonical_name, n.aliases AS aliases,
               n.embedding AS embedding, n.sourceDocuments AS source_documents,
               labels(n)[0] AS type,
               0.8 AS match_score
        LIMIT $limit
        
        ORDER BY match_score DESC
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, name=name, limit=limit)
            
            return [{
                "id": record["id"],
                "name": record["name"],
                "description": record["description"],
                "canonical_name": record["canonical_name"],
                "aliases": record["aliases"],
                "embedding": record["embedding"],
                "source_documents": record["source_documents"],
                "type": record["type"],
                "match_score": record["match_score"]
            } for record in result]
    
    def get_concepts_by_name(self, name, limit=10):
        """通过名称查找概念（兼容旧版方法）
        
        Args:
            name: 概念名称
            limit: 返回结果数量限制
            
        Returns:
            list: 概念列表
        """
        # 调用新方法，指定只查找Concept类型
        nodes = self.get_nodes_by_name(name, node_type='Concept', limit=limit)
        # 移除type字段以保持向后兼容
        return [{k: v for k, v in node.items() if k != 'type'} for node in nodes]
        
    def get_node_statistics(self):
        """获取节点统计信息
        
        Returns:
            dict: 包含Concept和Entity类型节点数量的字典
        """
        query = """
        // 获取Concept节点数量
        MATCH (c:Concept)
        RETURN count(c) AS concept_count
        
        UNION
        
        // 获取Entity节点数量
        MATCH (e:Entity)
        RETURN count(e) AS entity_count
        """
        
        with self.driver.session() as session:
            results = session.run(query)
            stats = {"concept_count": 0, "entity_count": 0}
            
            for record in results:
                if "concept_count" in record:
                    stats["concept_count"] = record["concept_count"]
                elif "entity_count" in record:
                    stats["entity_count"] = record["entity_count"]
            
            stats["total_count"] = stats["concept_count"] + stats["entity_count"]
            return stats
    
    def get_nodes_by_type(self, node_type, limit=100, skip=0):
        """根据类型获取节点
        
        Args:
            node_type: 节点类型 ('Concept' 或 'Entity')
            limit: 返回结果数量限制
            skip: 跳过的记录数
            
        Returns:
            list: 节点列表
        """
        if node_type not in ['Concept', 'Entity']:
            raise ValueError("node_type必须是'Concept'或'Entity'")
        
        query = f"""
        MATCH (n:{node_type})
        RETURN n.id AS id, n.name AS name, n.description AS description, 
               n.canonicalName AS canonical_name, n.aliases AS aliases,
               n.embedding AS embedding, n.sourceDocuments AS source_documents,
               labels(n)[0] AS type
        ORDER BY n.name
        SKIP $skip
        LIMIT $limit
        """
        
        with self.driver.session() as session:
            result = session.run(query, limit=limit, skip=skip)
            
            return [{
                "id": record["id"],
                "name": record["name"],
                "description": record["description"],
                "canonical_name": record["canonical_name"],
                "aliases": record["aliases"],
                "embedding": record["embedding"],
                "source_documents": record["source_documents"],
                "type": record["type"]
            } for record in result]
    
    def update_node_type(self, node_id, new_type):
        """更新节点类型
        
        Args:
            node_id: 节点ID
            new_type: 新的节点类型 ('Concept' 或 'Entity')
            
        Returns:
            bool: 是否更新成功
        """
        if new_type not in ['Concept', 'Entity']:
            raise ValueError("new_type必须是'Concept'或'Entity'")
        
        query = """
        MATCH (n) 
        WHERE n.id = $node_id AND (n:Concept OR n:Entity)
        REMOVE n:Concept, n:Entity
        SET n:$new_type
        RETURN count(n) AS count
        """
        
        with self.driver.session() as session:
            result = session.run(query, node_id=node_id, new_type=new_type)
            record = result.single()
            return record["count"] == 1
    
    def close(self):
        """关闭数据库连接"""
        if self.driver:
            self.driver.close()
        logger.info("Database connection closed")
    
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
                   n.canonicalName AS canonical_name, similarity
            ORDER BY similarity DESC
            """
            results = list(session.run(query, vector=query_vector.tolist(), 
                                      top_k=top_k, threshold=threshold))
            
            return [{
                'id': record['id'],
                'name': record['name'],
                'description': record['description'],
                'canonical_name': record['canonical_name'],
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
                   n.canonicalName AS canonical_name, similarity
            ORDER BY similarity DESC
            """
            results = list(session.run(query, vector=query_vector.tolist(), 
                                      top_k=top_k, threshold=threshold))
            
            return [{
                'id': record['id'],
                'name': record['name'],
                'description': record['description'],
                'canonical_name': record['canonical_name'],
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
    
    def search_knowledge_by_text(self, query_text, top_k=5):
        """
        基于文本搜索相关知识
        
        Args:
            query_text: 查询文本
            top_k: 返回的最大结果数
            
        Returns:
            组合的相关知识列表
        """
        # 获取查询向量
        query_vector = self.model.encode(query_text)
        
        # 搜索相关概念和实体
        concepts = self.search_related_concepts(query_vector, top_k)
        entities = self.search_related_entities(query_vector, top_k)
        
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
        
        # 为每个结果添加关系信息
        for result in all_results[:top_k]:
            relations = self.get_node_relations(result['id'], result['node_type'])
            result['relations'] = relations
        
        return all_results[:top_k]
