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
        
        # 使用本地缓存加载模型
        self.model = SentenceTransformer('all-MiniLM-L6-v2', 
                                      cache_folder=model_cache_dir,
                                      local_files_only=True)
        self.nlp = spacy.load("en_core_web_sm")  # 用于文本预处理
        
        # 初始化数据库索引
        self.__initialize_database_indexes()
    
    def __initialize_database_indexes(self):
        """创建必要的数据库索引"""
        with self.driver.session() as session:
            # 创建文本索引 - 分别为每个属性创建索引
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
            
            # 检查并创建向量索引（如果不存在）
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
            
            # 创建关系索引
            session.run("""
            CREATE INDEX rel_index IF NOT EXISTS 
            FOR ()-[r:RELATED_TO]-() ON (r.type)
            """)
            
            logger.info("Database indexes initialized")
    
    def __generate_vector(self, text):
        """为文本生成嵌入向量"""
        return self.model.encode([text])[0].tolist()
    
    def __find_similar_concepts(self, concept):
        """在Neo4j中查找相似概念"""
        query = """
        // 第一步: 名称精确匹配
        MATCH (c:Concept)
        WHERE c.canonicalName = $canonical_name 
           OR $name IN c.aliases
        RETURN c.id AS concept_id, c.name AS name, 1.0 AS similarity
        LIMIT 1
        
        UNION
        
        // 第二步: 名称模糊匹配
        MATCH (c:Concept)
        WHERE c.canonicalName CONTAINS $canonical_name 
           OR ANY(alias IN c.aliases WHERE alias CONTAINS $name)
        RETURN c.id AS concept_id, c.name AS name, 0.8 AS similarity
        LIMIT 5
        
        UNION
        
        // 第三步: 向量相似度搜索
        CALL db.index.vector.queryNodes('concept_vector_index', 10, $vector)
        YIELD node AS c, score AS similarity
        WHERE similarity > $threshold
        RETURN c.id AS concept_id, c.name AS name, similarity
        """
        
        params = {
            "name": concept["name"],
            "canonical_name": concept["canonical_name"],
            "vector": concept["vector"],
            "threshold": 0.7
        }
        
        with self.driver.session() as session:
            result = session.run(query, params)
            return [record for record in result]
    
    def __merge_or_create_concept(self, concept, candidates):
        """根据相似度决定合并或创建新概念"""
        # 如果没有候选，直接创建新概念
        if not candidates:
            return self.__create_new_concept(concept)
        
        # 选择最佳候选
        best_candidate = max(candidates, key=lambda x: x["similarity"])
        
        # 高相似度直接合并
        if best_candidate["similarity"] > 0.9:
            return self.__merge_concept(best_candidate["concept_id"], concept)
        
        # 中等相似度进行上下文分析
        context_similarity = self.__check_context_similarity(
            best_candidate["concept_id"], 
            concept["context"]
        )
        
        # 综合相似度
        combined_similarity = 0.7 * best_candidate["similarity"] + 0.3 * context_similarity
        
        if combined_similarity > 0.8:
            return self.__merge_concept(best_candidate["concept_id"], concept)
        else:
            return self.__create_new_concept(concept)
    
    def __check_context_similarity(self, concept_id, context):
        """检查概念上下文相似度"""
        query = """
        MATCH (c:Concept {id: $concept_id})-[:RELATED_TO]->(other:Concept)
        WITH c, collect(other.name) AS related_concepts
        
        RETURN size([name IN related_concepts WHERE name IN $context]) * 1.0 
               / size(related_concepts) AS context_similarity
        """
        
        with self.driver.session() as session:
            result = session.run(query, concept_id=concept_id, context=context)
            record = result.single()
            return record["context_similarity"] if record else 0
    
    def __create_new_concept(self, concept):
        """创建新概念节点"""
        query = """
        CREATE (c:Concept {
            id: apoc.create.uuid(),
            name: $name,
            canonicalName: $canonical_name,
            aliases: [$name],
            description: $description,
            embedding: $vector,
            sourceDocuments: [$source_document]
        })
        RETURN c.id AS concept_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                name=concept["name"],
                canonical_name=concept["canonical_name"],
                description=concept["description"],
                vector=concept["vector"],
                source_document=concept["source_document"]
            )
            record = result.single()
            return record["concept_id"]
    
    def __merge_concept(self, existing_id, new_concept):
        """合并到现有概念节点"""
        query = """
        MATCH (c:Concept {id: $existing_id})
        SET c.aliases = CASE WHEN NOT $new_name IN c.aliases 
                             THEN c.aliases + $new_name 
                             ELSE c.aliases END
        SET c.description = c.description + '\n' + $new_description
        SET c.sourceDocuments = c.sourceDocuments + $source_document
        RETURN c.id AS concept_id
        """
        
        with self.driver.session() as session:
            result = session.run(query, 
                existing_id=existing_id,
                new_name=new_concept["name"],
                new_description=new_concept["description"],
                source_document=new_concept["source_document"]
            )
            record = result.single()
            return record["concept_id"]
    
    def insert_concepts(self, concepts):
        """插入概念到知识图谱
        
        Args:
            concepts (list): 概念列表，每个概念为字典，包含name、description
                           可选字段：context（上下文列表）和source_document（来源文档）
            
        Returns:
            int: 成功处理的概念数量
        """
        processed_count = 0
        new_concepts_count = 0  # 新增概念计数
        merged_concepts_count = 0  # 合并概念计数
        
        # 处理每个概念
        for concept in concepts:
            try:
                # 验证必要字段
                if 'name' not in concept or 'description' not in concept:
                    logger.warning(f"概念缺少必要字段: {concept}")
                    continue
                
                # 1. 计算词向量
                concept_vector = self.__generate_vector(concept['name'])
                
                # 准备完整的概念数据，包含context和source_document
                full_concept = {
                    'name': concept['name'],
                    'canonical_name': self.nlp(concept['name'])[0].lemma_.lower() if self.nlp(concept['name']) else concept['name'].lower(),
                    'description': concept['description'],
                    'vector': concept_vector,
                    'context': concept.get('context', []),  # 使用概念中提供的上下文，默认为空列表
                    'source_document': concept.get('source_document', 'manual_insert')  # 使用概念中提供的来源文档，默认为manual_insert
                }
                
                # 2. 查找相似概念
                similar_concepts = self.__find_similar_concepts(full_concept)
                
                # 3. 合并或创建新概念
                concept_id = self.__merge_or_create_concept(full_concept, similar_concepts)
                
                # 统计新增和合并的概念
                if similar_concepts and any(c['similarity'] > 0.8 for c in similar_concepts):
                    merged_concepts_count += 1
                    logger.info(f"已合并概念 '{concept['name']}' -> ID: {concept_id}")
                else:
                    new_concepts_count += 1
                    logger.info(f"已新增概念 '{concept['name']}' -> ID: {concept_id}")
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"处理概念 '{concept.get('name', '未知')}' 时出错: {str(e)}")
                continue
        
        logger.info(f"概念处理完成: 共处理 {processed_count}/{len(concepts)} 个概念，其中新增 {new_concepts_count} 个，合并 {merged_concepts_count} 个")
        return processed_count
    
    def insert_relations(self, relations):
        """插入关系到知识图谱
        
        Args:
            relations (list): 关系列表，每个关系为字典，包含以下字段：
                            - source: 源概念名称
                            - target: 目标概念名称
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
                    
                    # 构建概念对象以使用现有的__find_similar_concepts方法
                    source_concept = {
                        'name': relation['source'],
                        'canonical_name': self.nlp(relation['source'])[0].lemma_.lower() if self.nlp(relation['source']) else relation['source'].lower(),
                        'vector': self.__generate_vector(relation['source'])
                    }
                    
                    target_concept = {
                        'name': relation['target'],
                        'canonical_name': self.nlp(relation['target'])[0].lemma_.lower() if self.nlp(relation['target']) else relation['target'].lower(),
                        'vector': self.__generate_vector(relation['target'])
                    }
                    
                    # 使用现有的__find_similar_concepts方法查找相似概念
                    source_candidates = self.__find_similar_concepts(source_concept)
                    target_candidates = self.__find_similar_concepts(target_concept)
                    
                    # 过滤并获取相似度最高的概念
                    source_candidates = [c for c in source_candidates if c['similarity'] >= similarity_threshold]
                    target_candidates = [c for c in target_candidates if c['similarity'] >= similarity_threshold]
                    
                    if not source_candidates:
                        logger.warning(f"找不到与'{relation['source']}'相似的概念（相似度阈值: {similarity_threshold}）")
                        continue
                    
                    if not target_candidates:
                        logger.warning(f"找不到与'{relation['target']}'相似的概念（相似度阈值: {similarity_threshold}）")
                        continue
                    
                    # 获取相似度最高的概念
                    best_source = max(source_candidates, key=lambda x: x['similarity'])
                    best_target = max(target_candidates, key=lambda x: x['similarity'])
                    
                    source_id = best_source['concept_id']
                    actual_source_name = best_source['name']
                    target_id = best_target['concept_id']
                    actual_target_name = best_target['name']
                    
                    # 检查现有关系
                    existing_relations = self.__check_existing_relations(session, source_id, target_id)
                    
                    if existing_relations:
                        # 关系已存在，检查是否相似
                        relation_exists = False
                        for existing_rel in existing_relations:
                            if self.__verify_relation_similarity(existing_rel, rel_type, properties):
                                # 关系相似，合并权重
                                self.__merge_relation_weights(session, existing_rel['id'], weight, properties)
                                logger.info(f"已合并相似关系: '{actual_source_name}' -[{rel_type}]-> '{actual_target_name}' (ID: {existing_rel['id']})")
                                relation_exists = True
                                merged_relations_count += 1
                                processed_count += 1
                                break
                        
                        if not relation_exists:
                            # 关系不相似，创建新关系
                            new_rel_id = self.__create_new_relation(session, source_id, target_id, rel_type, weight, properties)
                            logger.info(f"已创建不同类型关系: '{actual_source_name}' -[{rel_type}]-> '{actual_target_name}' (ID: {new_rel_id})")
                            new_relations_count += 1
                            processed_count += 1
                    else:
                        # 关系不存在，创建新关系
                        new_rel_id = self.__create_new_relation(session, source_id, target_id, rel_type, weight, properties)
                        logger.info(f"已创建新关系: '{actual_source_name}' -[{rel_type}]-> '{actual_target_name}' (ID: {new_rel_id})")
                        new_relations_count += 1
                        processed_count += 1
                        
                except Exception as e:
                    logger.error(f"处理关系时出错: {str(relation)} - {str(e)}")
                    continue
        
        logger.info(f"关系处理完成: 共处理 {processed_count}/{len(relations)} 个关系，其中新增 {new_relations_count} 个，合并 {merged_relations_count} 个")
        return processed_count
    
    def __check_existing_relations(self, session, source_id, target_id):
        """检查源概念和目标概念之间是否存在关系"""
        query = """
        MATCH (source:Concept {id: $source_id})-[r]->(target:Concept {id: $target_id})
        RETURN id(r) AS id, type(r) AS type, properties(r) AS properties
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
    
    def __merge_relation_weights(self, session, relation_id, weight, properties):
        """合并关系权重和更新属性"""
        params = {
            'relation_id': relation_id,
            'weight': weight,
            **properties
        }
        
        query = """
        MATCH ()-[r]->()
        WHERE id(r) = $relation_id
        SET r.weight = coalesce(r.weight, 0) + $weight,
            r.last_updated = datetime()
        """
        
        # 添加额外属性更新
        for key in properties:
            if key not in ['weight', 'last_updated']:
                query += f"\n        SET r.{key} = ${key}"
        
        session.run(query, params)
    
    def __create_new_relation(self, session, source_id, target_id, rel_type, weight, properties):
        """创建新关系"""
        params = {
            'source_id': source_id,
            'target_id': target_id,
            'weight': weight,
            **properties
        }
        
        query = f"""
        MATCH (source:Concept {{id: $source_id}})
        MATCH (target:Concept {{id: $target_id}})
        CREATE (source)-[r:{rel_type} {{weight: $weight, last_updated: datetime()}}]->(target)
        """
        
        # 添加额外属性
        for key, value in properties.items():
            if key not in ['weight', 'last_updated']:
                query += f"\n        SET r.{key} = ${key}"
        
        query += "\n        RETURN id(r) AS relation_id"
        
        result = session.run(query, params)
        record = result.single()
        return record['relation_id'] if record else None
    
    def close(self):
        """关闭数据库连接"""
        self.driver.close()
        logger.info("Database connection closed")


# 示例用法
if __name__ == "__main__":
    # 配置数据库连接
    NEO4J_URI = "neo4j://127.0.0.1:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "12345678"
    
    # 初始化知识图谱系统
    kg = Neo4jKnowledgeGraph(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    
    # 示例文本
    texts = [
        "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data.",
        "Deep learning, a specialized form of machine learning, uses neural networks with many layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret and understand visual information."
    ]
    
    # 处理文本
    kg.batch_process_texts(texts)
    
    # 关闭连接
    kg.close()